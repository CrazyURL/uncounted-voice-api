"""AI허브 저음질 전화망 음성인식 데이터로 Whisper large-v3 파인튜닝 스크립트

목적:
    전화망 특유의 8kHz 저음질·압축 노이즈 환경에서 Whisper 인식 정확도를 높인다.
    파인튜닝된 모델을 WhisperX 서버에 적용하면 실제 전화 녹음 파일의
    전체적인 음성 식별 능력이 향상된다.

데이터셋:
    AI허브 "저음질 전화망 음성인식 데이터"
    - 8kHz (또는 16kHz), 단채널, 전화망 노이즈·압축 열화
    - JSON 사이드카에 발화 전사 포함

사용법:
    # 구조 검증만 (실제 모델/데이터 불필요)
    python scripts/finetune_whisperx.py --dummy

    # 실제 학습 (WhisperX 서비스 중단 후 실행 권장 — VRAM 공유)
    export DATASET_AIHUB_TELEPHONE_DIR=/data/aihub/telephone
    export HF_TOKEN=hf_xxxx
    python scripts/finetune_whisperx.py \\
        --epochs 3 \\
        --batch_size 4 \\
        --lr 1e-5 \\
        --output_dir models/whisperx

WhisperX 적용 방법 (학습 완료 후):
    # 1) HuggingFace 포맷 → CTranslate2 포맷 변환 (faster-whisper 호환)
    pip install ctranslate2
    ct2-transformers-converter \\
        --model models/whisperx/current/hf \\
        --output_dir models/whisperx/current/ct2 \\
        --quantization float16

    # 2) env var 설정
    WHISPERX_MODEL_DIR=models/whisperx/current/ct2

    # 3) 서비스 재시작
    sudo systemctl restart voice-api@dev

환경변수:
    DATASET_AIHUB_TELEPHONE_DIR   저음질 전화망 데이터 루트 (선택)
    WHISPERX_BASE_MODEL           기반 모델 (기본값: openai/whisper-large-v3)
    HF_TOKEN                      HuggingFace 토큰 (모델 다운로드 시 필요)

출력:
    models/whisperx/v{YYYYMMDD_HHMMSS}/
        hf/                    — HuggingFace 포맷 파인튜닝 모델
        metrics.json           — 에포크별 train_loss / val_wer
        training_status.json   — 실시간 진행 상태 (routers/training.py 폴링용)
    models/whisperx/current.txt  — 최신 버전명 (Linux에서는 symlink도 생성)

제한:
    - 학습은 GPU(CUDA) 강력 권장. CPU 모드는 에포크당 수 시간 소요.
    - WhisperX 서비스와 동시 실행 시 VRAM OOM. 서비스 중단 후 실행할 것.
    - --dummy 모드는 합성 텍스트/오디오로 DataLoader 구조만 검증한다.
    - 변환(ct2-transformers-converter)은 별도 패키지 필요 — 이 스크립트 범위 밖.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Windows 콘솔 UTF-8 출력
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── 상수 ──────────────────────────────────────────────────────────────────────

_DEFAULT_BASE_MODEL = "openai/whisper-large-v3"
_TARGET_SAMPLE_RATE = 16_000        # Whisper 입력 샘플레이트
_MAX_AUDIO_SECONDS  = 30            # Whisper 컨텍스트 윈도우 (고정)
_MAX_AUDIO_SAMPLES  = _TARGET_SAMPLE_RATE * _MAX_AUDIO_SECONDS
_LOG_MEL_FRAMES     = 3000          # Whisper 스펙트로그램 프레임 수
_N_MELS             = 128           # large-v3 멜 필터 수
_EARLY_STOP_PATIENCE = 3            # val_wer 개선 없을 때 조기 종료 에포크 수
_MIN_SAMPLES_FOR_TRAIN = 50         # 더미 아닌 실제 학습 최소 샘플 수


# ── 학습 상태 ─────────────────────────────────────────────────────────────────

@dataclass
class TrainingStatus:
    job_id: str
    status: str          # "running" | "completed" | "failed"
    current_epoch: int
    total_epochs: int
    train_loss: float
    val_wer: float        # Word Error Rate (낮을수록 좋음)
    elapsed_sec: float
    model_version: str
    error_message: str = ""

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ── AI허브 전화망 데이터 로더 ──────────────────────────────────────────────────

def _find_wav_files(root: Path) -> list[Path]:
    """재귀적으로 WAV 파일을 수집한다."""
    return sorted(root.rglob("*.wav")) + sorted(root.rglob("*.WAV"))


def _load_transcript_from_json(json_path: Path) -> str:
    """JSON 사이드카에서 발화 전사문을 추출한다.

    AI허브 전화망 데이터는 여러 JSON 스키마를 사용하므로 우선순위 순으로 시도.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    # 후보 키 목록 (우선순위 순)
    for key in ("발화내용", "transcript", "text", "utterance", "sentence", "stt"):
        if key in data and isinstance(data[key], str):
            return data[key].strip()

    # dialogs / utterances 배열 구조
    for arr_key in ("dialogs", "utterances", "data"):
        arr = data.get(arr_key)
        if isinstance(arr, list) and arr:
            item = arr[0]
            if isinstance(item, dict):
                for key in ("발화내용", "transcript", "text", "utterance", "sentence"):
                    if key in item and isinstance(item[key], str):
                        return item[key].strip()

    return ""


def _find_transcript(wav_path: Path) -> str:
    """WAV 파일과 짝이 되는 전사문을 찾는다.

    탐색 순서:
    1. 동명 .json 사이드카
    2. 동명 .txt (UTF-8)
    3. 상위 label/ 디렉터리의 동명 .json / .txt
    """
    stem = wav_path.stem

    # 1) 동일 디렉터리 JSON
    json_sibling = wav_path.with_suffix(".json")
    if json_sibling.exists():
        t = _load_transcript_from_json(json_sibling)
        if t:
            return t

    # 2) 동일 디렉터리 TXT
    txt_sibling = wav_path.with_suffix(".txt")
    if txt_sibling.exists():
        try:
            return txt_sibling.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # 3) 상위 label/ 디렉터리
    for label_dir_name in ("label", "labels", "txt", "json", "transcript"):
        label_dir = wav_path.parent.parent / label_dir_name
        for ext in (".json", ".txt"):
            label_file = label_dir / (stem + ext)
            if label_file.exists():
                if ext == ".json":
                    t = _load_transcript_from_json(label_file)
                else:
                    try:
                        t = label_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        t = ""
                if t:
                    return t

    return ""


def load_telephone_dataset(root: Path) -> list[dict]:
    """AI허브 저음질 전화망 데이터셋을 읽어 {wav_path, transcript} 리스트를 반환한다."""
    wav_files = _find_wav_files(root)
    if not wav_files:
        logger.warning("WAV 파일 없음: %s", root)
        return []

    samples: list[dict] = []
    missing_transcript = 0
    for wav in wav_files:
        transcript = _find_transcript(wav)
        if not transcript:
            missing_transcript += 1
            continue
        samples.append({"wav_path": wav, "transcript": transcript})

    logger.info(
        "전화망 데이터: %d 샘플 로드 (전사 없음 제외: %d)",
        len(samples), missing_transcript,
    )
    return samples


# ── 더미 데이터 ────────────────────────────────────────────────────────────────

_DUMMY_SENTENCES = [
    "안녕하세요 반갑습니다",
    "오늘 날씨가 맑고 좋네요",
    "전화 연결 감사합니다",
    "잠시만 기다려 주십시오",
    "확인하고 바로 연락드리겠습니다",
    "죄송합니다 다시 한번 말씀해 주시겠어요",
    "네 알겠습니다 처리해 드리겠습니다",
    "고객님의 개인정보 보호를 위해 본인 확인이 필요합니다",
    "더 필요하신 사항이 있으시면 말씀해 주세요",
    "감사합니다 좋은 하루 되세요",
]


def create_dummy_samples(n: int = 40) -> list[dict]:
    """실제 파일 없이 구조 검증을 위한 더미 샘플을 생성한다.

    실제 오디오 대신 None을 사용하며, 더미 DataLoader가 가우시안 노이즈를 생성.
    """
    samples = []
    for i in range(n):
        samples.append({
            "wav_path": None,
            "transcript": _DUMMY_SENTENCES[i % len(_DUMMY_SENTENCES)],
        })
    return samples


# ── PyTorch Dataset ─────────────────────────────────────────────────────────

def _load_audio_numpy(wav_path: Path, target_sr: int) -> "np.ndarray":
    """WAV를 읽어 mono float32 배열로 반환한다 (최대 30초 클리핑).

    torchaudio로 로드·리샘플링하므로 soundfile/resampy 불필요.
    """
    import numpy as np
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(wav_path))   # (channels, samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    audio = waveform.squeeze(0).numpy()

    max_samples = target_sr * _MAX_AUDIO_SECONDS
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio.astype(np.float32)


def _compute_log_mel(audio: "np.ndarray", processor: "WhisperProcessor") -> "np.ndarray":
    """오디오 배열을 Whisper 입력 log-mel 스펙트로그램으로 변환한다."""
    features = processor(
        audio,
        sampling_rate=_TARGET_SAMPLE_RATE,
        return_tensors="np",
    )
    return features.input_features[0]   # (n_mels, n_frames)


class TelephoneAudioDataset:
    """저음질 전화망 오디오 PyTorch Dataset."""

    def __init__(
        self,
        samples: list[dict],
        processor: "WhisperProcessor",
        dummy: bool = False,
    ) -> None:
        self._samples = samples
        self._processor = processor
        self._dummy = dummy

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np
        import torch

        sample = self._samples[idx]
        transcript = sample["transcript"]

        if self._dummy or sample["wav_path"] is None:
            # 가우시안 노이즈로 8kHz 전화음 시뮬레이션 (1초)
            raw = np.random.randn(_TARGET_SAMPLE_RATE).astype(np.float32) * 0.02
        else:
            try:
                raw = _load_audio_numpy(sample["wav_path"], _TARGET_SAMPLE_RATE)
            except Exception as e:
                logger.debug("오디오 로드 실패 (%s): %s", sample["wav_path"], e)
                raw = np.zeros(_TARGET_SAMPLE_RATE, dtype=np.float32)

        input_features = _compute_log_mel(raw, self._processor)  # (n_mels, n_frames)

        labels = self._processor.tokenizer(
            transcript,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=448,   # Whisper decoder max tokens
        ).input_ids[0]

        return {
            "input_features": torch.from_numpy(input_features).unsqueeze(0),  # (1, n_mels, frames)
            "labels": torch.from_numpy(labels).long(),
            "transcript": transcript,
        }


def _collate_fn(batch: list[dict]) -> dict:
    """가변 길이 labels를 -100 패딩으로 배치 처리한다."""
    import torch

    input_features = torch.cat([b["input_features"] for b in batch], dim=0)

    max_len = max(b["labels"].size(0) for b in batch)
    padded_labels = []
    for b in batch:
        lbl = b["labels"]
        pad = torch.full((max_len - lbl.size(0),), -100, dtype=torch.long)
        padded_labels.append(torch.cat([lbl, pad], dim=0))

    return {
        "input_features": input_features,
        "labels": torch.stack(padded_labels, dim=0),
        "transcripts": [b["transcript"] for b in batch],
    }


# ── WER 계산 ───────────────────────────────────────────────────────────────────

def _levenshtein(a: list, b: list) -> int:
    """Levenshtein 거리 (리스트 원소 단위)."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """코퍼스 수준 WER (Word Error Rate)를 계산한다."""
    total_errors, total_words = 0, 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_errors += _levenshtein(ref_words, hyp_words)
        total_words  += max(len(ref_words), 1)
    return total_errors / max(total_words, 1)


# ── Whisper 파인튜너 ───────────────────────────────────────────────────────────

class WhisperFineTuner:
    """Whisper large-v3 파인튜너.

    - GPU 사용 시 WhisperX 서비스 중단 후 실행 권장 (VRAM 공유).
    - 더미 모드: tiny 모델로 DataLoader 구조 검증만 수행.
    """

    def __init__(
        self,
        base_model: str,
        output_dir: Path,
        dummy: bool = False,
    ) -> None:
        self._base_model = "openai/whisper-tiny" if dummy else base_model
        self._output_dir = output_dir
        self._dummy = dummy
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError:
            raise RuntimeError(
                "transformers 패키지가 필요합니다: pip install 'transformers>=4.57.6'"
            )

        hf_token = os.environ.get("HF_TOKEN")
        logger.info("Whisper 모델 로드: %s", self._base_model)
        self._processor = WhisperProcessor.from_pretrained(
            self._base_model,
            token=hf_token,
            language="korean",
            task="transcribe",
        )
        self._processor.tokenizer.set_prefix_tokens(language="korean", task="transcribe")
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self._base_model,
            token=hf_token,
        )
        logger.info("모델 로드 완료")

    @property
    def processor(self) -> "WhisperProcessor":
        self._load()
        return self._processor  # type: ignore[return-value]

    def train(
        self,
        train_samples: list[dict],
        val_samples: list[dict],
        epochs: int,
        batch_size: int,
        lr: float,
        status: TrainingStatus,
        status_path: Path,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader

        self._load()
        model = self._model
        processor = self._processor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("학습 디바이스: %s", device)
        model = model.to(device)

        train_ds = TelephoneAudioDataset(train_samples, processor, dummy=self._dummy)
        val_ds   = TelephoneAudioDataset(val_samples,   processor, dummy=self._dummy)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=0,          # Windows 호환성 + GPU 단일 프로세스
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, batch_size // 2),
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # 선형 LR 감쇠 스케줄러
        total_steps = epochs * max(len(train_loader), 1)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps,
        )

        best_wer = float("inf")
        no_improve_epochs = 0
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            # ── 학습 ──
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                feats  = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_features=feats, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # ── 검증 ──
            model.eval()
            refs, hyps = [], []
            with torch.no_grad():
                for batch in val_loader:
                    feats = batch["input_features"].to(device)
                    forced_ids = processor.get_decoder_prompt_ids(
                        language="korean", task="transcribe"
                    )
                    predicted_ids = model.generate(
                        feats,
                        forced_decoder_ids=forced_ids,
                        max_new_tokens=448,
                    )
                    decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                    refs.extend(batch["transcripts"])
                    hyps.extend(decoded)

            val_wer = compute_wer(refs, hyps)

            elapsed = time.time() - t0
            logger.info(
                "에포크 %d/%d | loss=%.4f | WER=%.4f | %.0fs",
                epoch, epochs, avg_loss, val_wer, elapsed,
            )

            # 상태 파일 갱신
            status.current_epoch = epoch
            status.train_loss    = round(avg_loss, 6)
            status.val_wer       = round(val_wer, 6)
            status.elapsed_sec   = round(elapsed, 1)
            status.save(status_path)

            # 최적 모델 체크포인트
            if val_wer < best_wer:
                best_wer = val_wer
                no_improve_epochs = 0
                self._save_checkpoint(epoch, avg_loss, val_wer)
            else:
                no_improve_epochs += 1
                logger.info("WER 개선 없음 (%d/%d)", no_improve_epochs, _EARLY_STOP_PATIENCE)
                if no_improve_epochs >= _EARLY_STOP_PATIENCE:
                    logger.info("조기 종료 (WER 연속 %d에포크 미개선)", _EARLY_STOP_PATIENCE)
                    break

        logger.info("학습 완료. 최적 WER=%.4f", best_wer)

    def _save_checkpoint(self, epoch: int, loss: float, wer: float) -> None:
        """현재 최적 모델을 HuggingFace 포맷으로 저장한다."""
        save_dir = self._output_dir / "hf"
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(save_dir))
        self._processor.save_pretrained(str(save_dir))
        logger.info(
            "체크포인트 저장: epoch=%d loss=%.4f wer=%.4f -> %s",
            epoch, loss, wer, save_dir,
        )


# ── 버전 관리 ──────────────────────────────────────────────────────────────────

def _make_version_dir(output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = output_dir / f"v{ts}"
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def _update_current_pointer(output_dir: Path, version_name: str) -> None:
    """current.txt 를 항상 쓰고, Linux에서는 symlink도 생성한다."""
    txt_ptr = output_dir / "current.txt"
    txt_ptr.write_text(version_name, encoding="utf-8")

    link = output_dir / "current"
    target = output_dir / version_name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)
        logger.info("symlink 갱신: %s -> %s", link, target)
    except (OSError, NotImplementedError):
        logger.debug("symlink 생성 불가 (Windows 제한) — current.txt만 사용")


# ── 데이터 분할 ────────────────────────────────────────────────────────────────

def _split_samples(
    samples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """샘플을 train / val 로 분할한다."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


# ── 지표 저장 ─────────────────────────────────────────────────────────────────

def _save_metrics(
    version_dir: Path,
    n_train: int,
    n_val: int,
    base_model: str,
    final_status: TrainingStatus,
) -> None:
    metrics = {
        "base_model": base_model,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "final_train_loss": final_status.train_loss,
        "final_val_wer": final_status.val_wer,
        "total_epochs_run": final_status.current_epoch,
        "elapsed_sec": final_status.elapsed_sec,
        "completed_at": datetime.now().isoformat(),
    }
    (version_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper large-v3 저음질 전화망 파인튜닝")
    p.add_argument("--dummy",        action="store_true",
                   help="더미 데이터 + tiny 모델로 구조 검증만 수행")
    p.add_argument("--base_model",   default=_DEFAULT_BASE_MODEL,
                   help=f"HuggingFace 베이스 모델 (기본값: {_DEFAULT_BASE_MODEL})")
    p.add_argument("--output_dir",   default="models/whisperx",
                   help="모델 출력 루트 디렉터리 (기본값: models/whisperx)")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--val_ratio",    type=float, default=0.1,
                   help="검증 셋 비율 (기본값: 0.1)")
    p.add_argument("--job_id",       default=None,
                   help="외부에서 주입하는 job ID (routers/training.py 연동용)")
    return p.parse_args()


def _check_dependencies(dummy: bool) -> None:
    """필수 패키지 설치 여부를 사전 확인한다."""
    missing = []
    for pkg in ("torch", "transformers"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # torchaudio는 실제 오디오 로드 시 필요 (torch가 있으면 보통 함께 설치됨)
    if not dummy:
        try:
            __import__("torchaudio")
        except ImportError:
            missing.append("torchaudio")

    if missing:
        joined = ", ".join(missing)
        logger.error(
            "필수 패키지 누락: %s\n"
            "  pip install %s",
            joined, joined,
        )
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    job_id      = args.job_id or str(uuid.uuid4())
    base_model  = args.base_model
    output_root = Path(args.output_dir)
    dummy       = args.dummy

    _check_dependencies(dummy)

    # ── 버전 디렉터리 생성 ──
    version_dir  = _make_version_dir(output_root)
    version_name = version_dir.name
    status_path  = version_dir / "training_status.json"

    status = TrainingStatus(
        job_id        = job_id,
        status        = "running",
        current_epoch = 0,
        total_epochs  = args.epochs,
        train_loss    = 0.0,
        val_wer       = 1.0,
        elapsed_sec   = 0.0,
        model_version = version_name,
    )
    status.save(status_path)

    try:
        # ── 데이터 로드 ──
        if dummy:
            logger.info("[더미 모드] 합성 샘플 40개 생성")
            all_samples = create_dummy_samples(40)
        else:
            telephone_dir = os.environ.get("DATASET_AIHUB_TELEPHONE_DIR")
            if not telephone_dir:
                logger.info(
                    "DATASET_AIHUB_TELEPHONE_DIR 미설정 — 환경변수를 설정한 후 재실행하세요.\n"
                    "  export DATASET_AIHUB_TELEPHONE_DIR=/path/to/aihub/telephone"
                )
                status.status = "failed"
                status.error_message = "DATASET_AIHUB_TELEPHONE_DIR not set"
                status.save(status_path)
                return

            all_samples = load_telephone_dataset(Path(telephone_dir))

            if len(all_samples) < _MIN_SAMPLES_FOR_TRAIN:
                logger.warning(
                    "샘플 수 부족 (%d < %d). --dummy 플래그 또는 데이터 경로를 확인하세요.",
                    len(all_samples), _MIN_SAMPLES_FOR_TRAIN,
                )
                status.status = "failed"
                status.error_message = f"insufficient samples: {len(all_samples)}"
                status.save(status_path)
                return

        train_samples, val_samples = _split_samples(all_samples, val_ratio=args.val_ratio)
        logger.info("train=%d  val=%d", len(train_samples), len(val_samples))

        # ── 파인튜너 초기화 및 학습 ──
        fine_tuner = WhisperFineTuner(
            base_model = base_model,
            output_dir = version_dir,
            dummy      = dummy,
        )

        fine_tuner.train(
            train_samples = train_samples,
            val_samples   = val_samples,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            lr            = args.lr,
            status        = status,
            status_path   = status_path,
        )

        # ── 지표 저장 ──
        _save_metrics(
            version_dir = version_dir,
            n_train     = len(train_samples),
            n_val       = len(val_samples),
            base_model  = "openai/whisper-tiny" if dummy else base_model,
            final_status= status,
        )

        # ── 버전 포인터 갱신 ──
        _update_current_pointer(output_root, version_name)

        status.status = "completed"
        status.save(status_path)

        # ── 완료 안내 ──
        logger.info("=" * 60)
        logger.info("Whisper 파인튜닝 완료")
        logger.info("  버전  : %s", version_name)
        logger.info("  WER   : %.4f", status.val_wer)
        logger.info("  출력  : %s/hf/", version_dir)
        logger.info("")
        logger.info("WhisperX 적용 방법:")
        logger.info("  1) ct2 변환:")
        logger.info(
            "     ct2-transformers-converter --model %s/hf --output_dir %s/ct2 --quantization float16",
            version_dir, version_dir,
        )
        logger.info("  2) 환경변수 설정:")
        logger.info("     WHISPERX_MODEL_DIR=%s/ct2", version_dir)
        logger.info("  3) 서비스 재시작:")
        logger.info("     sudo systemctl restart voice-api@dev")
        logger.info("=" * 60)

    except Exception as exc:
        logger.exception("학습 중 오류 발생: %s", exc)
        status.status = "failed"
        status.error_message = str(exc)
        status.save(status_path)
        sys.exit(1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
