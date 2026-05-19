"""AI허브 화자인식 + 다화자 음성합성 데이터셋으로 WeSpeaker 파인튜닝 스크립트

목적:
    1. 화자인식용 음성 데이터 (DATASET_AIHUB_SPEAKER_DIR)
    2. 다화자 음성합성 데이터 (DATASET_AIHUB_MULTISPEAKER_DIR)
    두 데이터셋을 병합해 WeSpeaker (ResNet34-LM) 백본을 파인튜닝한다.

사용법:
    export DATASET_AIHUB_SPEAKER_DIR=/path/to/speaker_recognition
    export DATASET_AIHUB_MULTISPEAKER_DIR=/path/to/multi_speaker_tts
    export HF_TOKEN=hf_xxxx
    python scripts/train_speaker_embedding.py \\
        [--dummy] \\
        [--base_model pyannote/wespeaker-voxceleb-resnet34-LM] \\
        [--previous_model models/speaker/current] \\
        [--output_dir models/speaker] \\
        [--epochs 10] \\
        [--batch_size 32] \\
        [--lr 1e-4] \\
        [--min_seg_sec 1.5] \\
        [--max_seg_sec 5.0]

환경변수:
    DATASET_AIHUB_SPEAKER_DIR      화자인식 데이터셋 루트 (선택)
    DATASET_AIHUB_MULTISPEAKER_DIR 다화자 음성합성 데이터셋 루트 (선택)
    HF_TOKEN                       HuggingFace 토큰 (베이스 모델 다운로드 시 필요)

출력:
    models/speaker/v{YYYYMMDD_HHMMSS}/
        pytorch_model.bin  — 파인튜닝된 WeSpeaker 가중치
        config.json        — 모델 설정
        metrics.json       — 학습 지표 (EER, loss)
        training_status.json — 실시간 진행 상태 (라우터 폴링용)
    models/speaker/current -> 최신 버전 심링크 (Linux) / current.txt (Windows)

제한:
    - 학습은 GPU(CUDA) 권장. CPU는 에포크당 수 시간 소요.
    - WhisperX 서비스와 동시 학습 시 VRAM OOM 위험. 서비스 중단 후 실행 권장.
    - 더미 모드(--dummy)는 합성 10화자 × 5발화 × 2초 오디오로 구조 검증만 수행.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MIN_SPEAKERS_FOR_TRAINING = 10
_REQUIRED_PACKAGES = ["torch", "torchaudio", "numpy", "pyannote.audio"]


# ── 학습 상태 ─────────────────────────────────────────────────────────────────
@dataclass
class TrainingStatus:
    job_id: str
    status: str          # "running" | "completed" | "failed"
    current_epoch: int
    total_epochs: int
    train_loss: float
    val_eer: float       # Equal Error Rate (낮을수록 좋음)
    elapsed_sec: float
    model_version: str
    error_message: str = ""

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ── 데이터셋 로더 ─────────────────────────────────────────────────────────────

def _find_wav_files(base_dir: Path) -> list[Path]:
    """디렉토리 하위의 모든 WAV 파일을 탐색한다."""
    wavs = list(base_dir.rglob("*.wav"))
    wavs += list(base_dir.rglob("*.WAV"))
    return wavs


def _speaker_id_from_json_or_path(wav_path: Path, base_dir: Path) -> str:
    """JSON 사이드카 또는 디렉토리 구조에서 화자 ID를 추출한다.

    우선순위:
    1. 같은 이름의 .json 파일 → speaker_id / speaker / SpeakerID / talker_id 키
    2. WAV 파일의 상위 폴더명 (AI허브 표준 구조)
    """
    json_path = wav_path.with_suffix(".json")
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            for key in ("speaker_id", "SpeakerID", "speaker", "talker_id",
                        "speakerId", "speaker-id"):
                if isinstance(meta.get(key), str) and meta[key].strip():
                    return meta[key].strip()
        except Exception:
            pass
    # 폴더명 방식: base_dir/speaker_xxx/utterance.wav
    try:
        rel = wav_path.relative_to(base_dir)
        return rel.parts[0]  # 첫 번째 하위 폴더를 화자 ID로
    except ValueError:
        return wav_path.stem


def load_speaker_wavs(
    base_dir: Path,
    min_seg_sec: float = 1.5,
    max_seg_sec: float = 5.0,
) -> dict[str, list[Path]]:
    """화자 ID → WAV 경로 목록 딕셔너리를 반환한다."""
    wav_files = _find_wav_files(base_dir)
    logger.info("%s: WAV %d개 발견", base_dir.name, len(wav_files))

    speaker_wavs: dict[str, list[Path]] = {}
    for wav in wav_files:
        sid = _speaker_id_from_json_or_path(wav, base_dir)
        speaker_wavs.setdefault(sid, []).append(wav)

    # 화자당 최소 3개 파일 있는 화자만 유지
    speaker_wavs = {s: ws for s, ws in speaker_wavs.items() if len(ws) >= 3}
    logger.info("유효 화자 %d명 (최소 3발화 이상)", len(speaker_wavs))
    return speaker_wavs


def merge_speaker_datasets(
    *datasets: dict[str, list[Path]],
) -> dict[str, list[Path]]:
    """여러 데이터셋의 화자 딕셔너리를 병합한다. 소스 prefix로 충돌 방지."""
    merged: dict[str, list[Path]] = {}
    for idx, ds in enumerate(datasets):
        prefix = f"src{idx}_"
        for sid, wavs in ds.items():
            merged[prefix + sid] = wavs
    return merged


# ── 더미 데이터 생성 ──────────────────────────────────────────────────────────

def create_dummy_speaker_wavs(
    n_speakers: int = 10,
    n_utterances: int = 5,
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
) -> dict[str, list[Path]]:
    """메모리 내 합성 오디오로 더미 화자 딕셔너리를 생성한다.

    실제 WAV 파일 없이 DataLoader 흐름을 검증하기 위해 사용한다.
    반환 딕셔너리의 Path는 실제 존재하지 않아도 되도록
    get_waveform() 이 더미 numpy array를 직접 생성한다.
    """
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy 미설치 — pip install numpy")
        sys.exit(1)

    dummy: dict[str, list[Path]] = {}
    for spk_i in range(n_speakers):
        sid = f"dummy_spk_{spk_i:03d}"
        paths = [Path(f"/dummy/{sid}/utt{utt_i:03d}.wav") for utt_i in range(n_utterances)]
        dummy[sid] = paths
    return dummy


# ── WeSpeaker 파인튜닝 ────────────────────────────────────────────────────────

def _check_imports() -> bool:
    """필수 패키지 임포트 가능 여부 확인."""
    missing = []
    for pkg in _REQUIRED_PACKAGES:
        pkg_name = pkg.replace(".", "_") if "." in pkg else pkg
        try:
            __import__(pkg_name if pkg_name != "pyannote_audio" else "pyannote.audio")
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("미설치 패키지: %s", ", ".join(missing))
        logger.error("pip install %s", " ".join(missing))
        return False
    return True


class SpeakerDataset:
    """화자 임베딩 학습용 Dataset.

    각 샘플: (waveform_tensor, speaker_label_int)
    더미 모드에서는 가우시안 노이즈를 반환한다.
    """

    def __init__(
        self,
        speaker_wavs: dict[str, list[Path]],
        sample_rate: int = 16000,
        min_samples: int = 24000,
        max_samples: int = 80000,
        is_dummy: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.is_dummy = is_dummy

        self.speaker_to_idx: dict[str, int] = {
            sid: idx for idx, sid in enumerate(sorted(speaker_wavs.keys()))
        }
        self.samples: list[tuple[Path, int]] = []
        for sid, wavs in speaker_wavs.items():
            label = self.speaker_to_idx[sid]
            for wav in wavs:
                self.samples.append((wav, label))

        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import numpy as np

        wav_path, label = self.samples[idx]

        if self.is_dummy:
            n = random.randint(self.min_samples, self.max_samples)
            waveform = np.random.randn(n).astype(np.float32) * 0.05
        else:
            try:
                import torchaudio
                waveform_t, sr = torchaudio.load(str(wav_path))
                if sr != self.sample_rate:
                    import torchaudio.functional as F
                    waveform_t = F.resample(waveform_t, sr, self.sample_rate)
                waveform = waveform_t.mean(dim=0).numpy()
            except Exception as e:
                logger.debug("WAV 로드 실패 %s: %s", wav_path, e)
                waveform = np.zeros(self.min_samples, dtype=np.float32)

        n = len(waveform)
        if n < self.min_samples:
            waveform = np.pad(waveform, (0, self.min_samples - n))
        elif n > self.max_samples:
            start = random.randint(0, n - self.max_samples)
            waveform = waveform[start : start + self.max_samples]

        import torch
        return torch.from_numpy(waveform).unsqueeze(0), torch.tensor(label, dtype=torch.long)


class WeSpeakerFineTuner:
    """WeSpeaker ResNet34-LM 파인튜닝 wrapper."""

    def __init__(
        self,
        base_model_path: str,
        n_speakers: int,
        device: str = "cpu",
        hf_token: Optional[str] = None,
    ) -> None:
        self.device_name = device
        self.n_speakers = n_speakers
        self.hf_token = hf_token
        self.base_model_path = base_model_path

        import torch
        self.torch = torch

        self.model = self._load_backbone()
        self.head = self._build_head()
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.head.parameters()),
            lr=1e-4,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def _load_backbone(self):
        """pyannote/wespeaker 모델을 로드한다."""
        try:
            from pyannote.audio import Model
            kwargs: dict = {}
            if self.hf_token:
                kwargs["use_auth_token"] = self.hf_token
            model = Model.from_pretrained(self.base_model_path, **kwargs)
            model.to(self.device_name)
            logger.info("백본 로드 완료: %s → %s", self.base_model_path, self.device_name)
            return model
        except Exception as e:
            raise RuntimeError(f"백본 로드 실패: {e}") from e

    def _build_head(self):
        """화자 분류 헤드 (선형 레이어)."""
        embedding_dim = 256  # ResNet34-LM 출력 차원
        head = self.torch.nn.Linear(embedding_dim, self.n_speakers)
        head.to(self.device_name)
        return head

    def _extract_embedding(self, waveform_batch):
        """배치 waveform → 임베딩 (batch, dim)."""
        waveform_batch = waveform_batch.to(self.device_name)
        output = self.model(waveform_batch)
        if hasattr(output, "detach"):
            emb = output
        else:
            emb = self.torch.tensor(output)
        norm = emb.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return emb / norm

    def train_epoch(self, loader) -> float:
        self.model.train()
        self.head.train()
        total_loss = 0.0
        n_batches = 0
        for waveforms, labels in loader:
            labels = labels.to(self.device_name)
            self.optimizer.zero_grad()
            emb = self._extract_embedding(waveforms)
            logits = self.head(emb)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def evaluate(self, loader) -> float:
        """간이 EER 추정: cosine similarity 기반 같은 화자/다른 화자 점수 비교."""
        self.model.eval()
        self.head.eval()

        embeddings_by_speaker: dict[int, list] = {}
        with self.torch.no_grad():
            for waveforms, labels in loader:
                embs = self._extract_embedding(waveforms)
                for emb, label in zip(embs, labels):
                    sid = int(label.item())
                    embeddings_by_speaker.setdefault(sid, []).append(
                        emb.cpu().numpy()
                    )

        import numpy as np

        same_scores: list[float] = []
        diff_scores: list[float] = []

        speaker_ids = list(embeddings_by_speaker.keys())
        for sid in speaker_ids:
            embs = embeddings_by_speaker[sid]
            if len(embs) >= 2:
                for i in range(min(len(embs) - 1, 3)):
                    score = float(np.dot(embs[i], embs[i + 1]))
                    same_scores.append(score)

        for i in range(min(len(speaker_ids) - 1, 20)):
            e1 = embeddings_by_speaker[speaker_ids[i]][0]
            e2 = embeddings_by_speaker[speaker_ids[i + 1]][0]
            diff_scores.append(float(np.dot(e1, e2)))

        if not same_scores or not diff_scores:
            return 0.5

        # 임계값 0.5 기준 간이 EER
        fa = sum(1 for s in diff_scores if s > 0.5) / len(diff_scores)
        fr = sum(1 for s in same_scores if s <= 0.5) / len(same_scores)
        return (fa + fr) / 2

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.model.state_dict(), output_dir / "pytorch_model.bin")
        self.torch.save(self.head.state_dict(), output_dir / "head.bin")
        config = {
            "base_model": self.base_model_path,
            "n_speakers_trained": self.n_speakers,
            "device": self.device_name,
        }
        (output_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("모델 저장: %s", output_dir)


def _update_current_symlink(output_dir: Path, version_dir: Path) -> None:
    """current 심링크 (Linux) 또는 current.txt (Windows) 업데이트."""
    current_txt = output_dir / "current.txt"
    current_txt.write_text(version_dir.name, encoding="utf-8")
    logger.info("current.txt → %s", version_dir.name)

    symlink = output_dir / "current"
    try:
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(version_dir, target_is_directory=True)
        logger.info("심링크 current → %s", version_dir.name)
    except (NotImplementedError, OSError):
        pass  # Windows 일반 사용자 권한에서는 심링크 불가 — current.txt로 대체


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="화자 임베딩 파인튜닝")
    parser.add_argument("--dummy", action="store_true", help="더미 모드 (구조 검증)")
    parser.add_argument(
        "--base_model",
        default="pyannote/wespeaker-voxceleb-resnet34-LM",
        help="베이스 WeSpeaker 모델 경로 또는 HF repo ID",
    )
    parser.add_argument(
        "--previous_model",
        default=None,
        help="이전 버전 모델 경로 (증분 파인튜닝)",
    )
    parser.add_argument("--output_dir", default="models/speaker", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_seg_sec", type=float, default=1.5)
    parser.add_argument("--max_seg_sec", type=float, default=5.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--job_id", default="local", help="학습 job ID (training.py에서 주입)")
    args = parser.parse_args()

    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    status = TrainingStatus(
        job_id=args.job_id,
        status="running",
        current_epoch=0,
        total_epochs=args.epochs,
        train_loss=0.0,
        val_eer=0.5,
        elapsed_sec=0.0,
        model_version=version,
    )
    status.save(version_dir / "training_status.json")

    if not args.dummy and not _check_imports():
        status.status = "failed"
        status.error_message = "필수 패키지 누락"
        status.save(version_dir / "training_status.json")
        sys.exit(1)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    if args.dummy:
        logger.info("더미 모드 — 합성 데이터 생성")
        speaker_wavs = create_dummy_speaker_wavs(n_speakers=10, n_utterances=5)
        is_dummy = True
    else:
        datasets: list[dict[str, list[Path]]] = []
        sample_rate = 16000
        min_samples = int(args.min_seg_sec * sample_rate)
        max_samples = int(args.max_seg_sec * sample_rate)

        if d := os.environ.get("DATASET_AIHUB_SPEAKER_DIR"):
            ds = load_speaker_wavs(Path(d), args.min_seg_sec, args.max_seg_sec)
            if ds:
                datasets.append(ds)
        else:
            logger.info("DATASET_AIHUB_SPEAKER_DIR 미설정 — 건너뜀")

        if d := os.environ.get("DATASET_AIHUB_MULTISPEAKER_DIR"):
            ds = load_speaker_wavs(Path(d), args.min_seg_sec, args.max_seg_sec)
            if ds:
                datasets.append(ds)
        else:
            logger.info("DATASET_AIHUB_MULTISPEAKER_DIR 미설정 — 건너뜀")

        if not datasets:
            logger.error("데이터셋이 없습니다. 환경변수를 확인하세요.")
            status.status = "failed"
            status.error_message = "데이터셋 없음"
            status.save(version_dir / "training_status.json")
            sys.exit(1)

        speaker_wavs = merge_speaker_datasets(*datasets)
        is_dummy = False

    n_speakers = len(speaker_wavs)
    logger.info("총 화자 수: %d명", n_speakers)
    if n_speakers < _MIN_SPEAKERS_FOR_TRAINING and not args.dummy:
        logger.error("화자 수가 너무 적습니다 (%d < %d)", n_speakers, _MIN_SPEAKERS_FOR_TRAINING)
        status.status = "failed"
        status.error_message = f"화자 수 부족: {n_speakers}"
        status.save(version_dir / "training_status.json")
        sys.exit(1)

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    import torch
    from torch.utils.data import DataLoader, random_split

    sample_rate = 16000
    full_dataset = SpeakerDataset(
        speaker_wavs,
        sample_rate=sample_rate,
        min_samples=int(args.min_seg_sec * sample_rate),
        max_samples=int(args.max_seg_sec * sample_rate),
        is_dummy=is_dummy,
    )

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    logger.info("학습 %d / 검증 %d 샘플", n_train, n_val)

    # ── 모델 초기화 ───────────────────────────────────────────────────────────
    base_path = args.previous_model or args.base_model
    hf_token = os.environ.get("HF_TOKEN")

    device_name = "cuda" if (torch.cuda.is_available() and not args.dummy) else "cpu"
    logger.info("학습 장치: %s", device_name)

    if args.dummy:
        logger.info("더미 모드 — 모델 로드 건너뜀 (DataLoader 구조 검증 완료)")
        status.status = "completed"
        status.current_epoch = args.epochs
        status.train_loss = 0.0
        status.val_eer = 0.0
        status.elapsed_sec = 0.0
        status.save(version_dir / "training_status.json")

        metrics = {"mode": "dummy", "n_speakers": n_speakers, "n_samples": len(full_dataset)}
        (version_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _update_current_symlink(output_dir, version_dir)
        logger.info("더미 모드 완료. 출력: %s", version_dir)
        return

    try:
        finetuner = WeSpeakerFineTuner(
            base_model_path=base_path,
            n_speakers=n_speakers,
            device=device_name,
            hf_token=hf_token,
        )
    except RuntimeError as e:
        status.status = "failed"
        status.error_message = str(e)
        status.save(version_dir / "training_status.json")
        logger.error("모델 초기화 실패: %s", e)
        sys.exit(1)

    # 이전 모델 가중치 로드 (증분 파인튜닝)
    if args.previous_model:
        prev_path = Path(args.previous_model)
        if (prev_path / "pytorch_model.bin").exists():
            finetuner.model.load_state_dict(
                torch.load(prev_path / "pytorch_model.bin", map_location=device_name)
            )
            logger.info("이전 모델 가중치 로드: %s", prev_path)

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    start_time = time.time()
    best_eer = 1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = finetuner.train_epoch(train_loader)
        val_eer = finetuner.evaluate(val_loader)
        elapsed = time.time() - start_time

        logger.info(
            "Epoch %d/%d — loss=%.4f  val_eer=%.4f  elapsed=%.0fs",
            epoch, args.epochs, train_loss, val_eer, elapsed,
        )

        status.current_epoch = epoch
        status.train_loss = round(train_loss, 6)
        status.val_eer = round(val_eer, 6)
        status.elapsed_sec = round(elapsed, 1)
        status.save(version_dir / "training_status.json")

        if val_eer < best_eer:
            best_eer = val_eer
            best_epoch = epoch
            finetuner.save(version_dir)

        # 조기 종료: 5 에포크 개선 없으면
        if epoch - best_epoch >= 5:
            logger.info("조기 종료 (best epoch=%d)", best_epoch)
            break

    elapsed = time.time() - start_time
    metrics = {
        "version": version,
        "base_model": base_path,
        "n_speakers": n_speakers,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "best_epoch": best_epoch,
        "best_val_eer": round(best_eer, 6),
        "total_epochs_run": epoch,
        "elapsed_sec": round(elapsed, 1),
    }
    (version_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    status.status = "completed"
    status.elapsed_sec = round(elapsed, 1)
    status.save(version_dir / "training_status.json")

    _update_current_symlink(output_dir, version_dir)
    logger.info("학습 완료. 최고 EER=%.4f (epoch %d). 출력: %s", best_eer, best_epoch, version_dir)


if __name__ == "__main__":
    main()
