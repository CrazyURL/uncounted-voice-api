"""
WhisperX baseline WER/CER 측정 스크립트

저음질 전화망 음성 데이터셋에서 random sample 후 WhisperX로 전사,
레퍼런스 텍스트와 비교해 WER/CER을 계산한다.

사용법:
  python scripts/measure_whisperx_baseline.py [options]

  --data PATH           오디오 데이터셋 루트 (기본: $TELEPHONE_ASR_DATASET_PATH)
  --sample-ratio FLOAT  전체 중 샘플 비율 (기본: 0.05)
  --max-samples N       최대 샘플 수 (기본: 500)
  --model-size STR      whisperx 모델 크기 (기본: large-v2)
  --language STR        언어 코드 (기본: ko)
  --batch-size N        배치 크기 (기본: 8)
  --output PATH         결과 JSON 저장 경로 (기본: logs/baseline_whisperx_{date}.json)
  --seed N              랜덤 시드 (기본: 42)
  --dummy               더미 모드 (실제 오디오 없이 메트릭 구조 검증)

출력 JSON:
  {
    "wer": 0.123,
    "cer": 0.045,
    "n_samples": 500,
    "total_duration_sec": 1234.5,
    "model_size": "large-v2",
    "language": "ko",
    "timestamp": "20260518_120000"
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 텍스트 정규화
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s가-힣]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# WER / CER 계산 (편집거리 기반)
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = _normalize(ref).split()
    hyp_words = _normalize(hyp).split()
    if not ref_words:
        return 0.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    ref_chars = list(_normalize(ref).replace(" ", ""))
    hyp_chars = list(_normalize(hyp).replace(" ", ""))
    if not ref_chars:
        return 0.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ---------------------------------------------------------------------------
# 오디오/레퍼런스 쌍 수집
# ---------------------------------------------------------------------------

def _find_audio_ref_pairs(data_dir: Path) -> list[tuple[Path, str]]:
    """(audio_path, reference_text) 쌍을 수집.

    전화망 데이터셋 구조:
      data_dir/
        *.json  — 레퍼런스 텍스트 포함
        *.wav / *.flac / *.mp3
    JSON 파일 내 텍스트 필드: text, transcript, utterance 중 존재하는 것.
    """
    pairs: list[tuple[Path, str]] = []
    audio_exts = {".wav", ".flac", ".mp3", ".m4a"}

    for json_path in data_dir.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue

        # 단일 발화 JSON
        ref_text = (data.get("text") or data.get("transcript") or
                    data.get("utterance") or data.get("norm_text") or "")
        if not ref_text and "utterances" in data:
            # 대화 단위 JSON — utterances 목록
            for utt in data.get("utterances", []):
                utt_text = (utt.get("text") or utt.get("transcript") or "").strip()
                if not utt_text:
                    continue
                # 대응 오디오 파일 탐색 (같은 디렉토리)
                stem = json_path.stem
                for ext in audio_exts:
                    candidate = json_path.parent / f"{stem}{ext}"
                    if candidate.exists():
                        pairs.append((candidate, utt_text))
                        break
            continue

        if not ref_text:
            continue

        # 대응 오디오 파일 탐색
        stem = json_path.stem
        for ext in audio_exts:
            candidate = json_path.parent / f"{stem}{ext}"
            if candidate.exists():
                pairs.append((candidate, ref_text.strip()))
                break

    return pairs


# ---------------------------------------------------------------------------
# 더미 모드
# ---------------------------------------------------------------------------

def _dummy_result(args) -> dict:
    logger.info("더미 모드: 실제 오디오 없이 메트릭 구조 검증")
    return {
        "wer": 0.0,
        "cer": 0.0,
        "n_samples": 0,
        "total_duration_sec": 0.0,
        "model_size": args.model_size,
        "language": args.language,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "note": "dummy mode — no actual inference",
    }


# ---------------------------------------------------------------------------
# 메인 측정
# ---------------------------------------------------------------------------

def measure(args) -> dict:
    if args.dummy:
        return _dummy_result(args)

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 디렉토리 없음: {data_dir}")

    logger.info("오디오/레퍼런스 쌍 수집: %s", data_dir)
    pairs = _find_audio_ref_pairs(data_dir)
    logger.info("총 쌍 수: %d", len(pairs))

    if not pairs:
        raise ValueError("오디오/레퍼런스 쌍을 찾을 수 없음. 데이터셋 구조 확인 필요")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_sample = min(args.max_samples, max(1, int(len(pairs) * args.sample_ratio)))
    pairs = pairs[:n_sample]
    logger.info("샘플 수: %d", len(pairs))

    try:
        import whisperx
    except ImportError:
        raise ImportError("whisperx 미설치. pip install whisperx")

    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA 없음 — CPU 사용 (느림)")
    except ImportError:
        device = "cpu"

    logger.info("모델 로드: %s (device=%s)", args.model_size, device)
    model = whisperx.load_model(args.model_size, device, language=args.language)

    total_wer, total_cer, total_dur = 0.0, 0.0, 0.0
    n_valid = 0

    for i, (audio_path, ref_text) in enumerate(pairs):
        try:
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=args.batch_size)
            hyp_segments = result.get("segments", [])
            hyp_text = " ".join(s.get("text", "") for s in hyp_segments).strip()

            dur = len(audio) / 16000 if hasattr(audio, "__len__") else 0.0
            total_dur += dur

            wer = compute_wer(ref_text, hyp_text)
            cer = compute_cer(ref_text, hyp_text)
            total_wer += wer
            total_cer += cer
            n_valid += 1

            if (i + 1) % 50 == 0:
                logger.info("진행: %d/%d  WER=%.3f  CER=%.3f",
                            i + 1, len(pairs),
                            total_wer / n_valid, total_cer / n_valid)
        except Exception as e:
            logger.debug("샘플 처리 실패: %s — %s", audio_path.name, e)

    if n_valid == 0:
        raise ValueError("유효 샘플 0건 — 오디오 포맷 또는 경로 확인 필요")

    result_dict = {
        "wer": round(total_wer / n_valid, 4),
        "cer": round(total_cer / n_valid, 4),
        "n_samples": n_valid,
        "n_attempted": len(pairs),
        "total_duration_sec": round(total_dur, 1),
        "model_size": args.model_size,
        "language": args.language,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    logger.info("결과: WER=%.4f  CER=%.4f  샘플=%d", result_dict["wer"], result_dict["cer"], n_valid)
    return result_dict


def main() -> None:
    default_data = os.environ.get("TELEPHONE_ASR_DATASET_PATH", "")
    today = datetime.now().strftime("%Y%m%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=default_data)
    parser.add_argument("--sample-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--model-size", default="large-v2")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=f"logs/baseline_whisperx_{today}.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    if not args.dummy and not args.data:
        raise EnvironmentError("--data 또는 TELEPHONE_ASR_DATASET_PATH 필요")

    result = measure(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
