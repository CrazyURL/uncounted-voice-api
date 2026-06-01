import logging
import re
import subprocess
import threading
import os
import time
from pathlib import Path

import numpy as np
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from app import config
from app.core.job_store import job_store
from app.pii_masker import mask_pii, mask_segments
from app.services.audio_preprocessor import load_df_model, preprocess
from app.services.diarization_config import DiarizationConfig
from app.services.recluster_config import ReclusterConfig
from app.services.speaker_embedding import SpeakerEmbeddingModel
from app.services.speaker_recluster import (
    maybe_recluster_speakers,
    renumber_speakers_in_place,
)
from app.services.utterance_segmenter import segment as segment_utterances
from app.services.audio_pii_masker import find_pii_word_ranges, mask_audio_ranges
from app.services.audio_splitter import (
    extract_utterance_audio,
    mute_non_speaker,
    to_wav_bytes,
)
from app.services.chunk_utterance_emitter import emit_chunk_utterances

logger = logging.getLogger(__name__)

# 전역 모델 (앱 시작 시 1회 로딩)
_model = None
_align_model = None
_align_metadata = None
_diarize_model = None
_speaker_embedding_model = None  # Phase 7: WeSpeaker embedding (lazy-loaded)
_gpu_lock = threading.Semaphore(1)  # GPU 동시 1건만 추론


# ---------------------------------------------------------------------------
# 대용량 오디오 청크 분할 헬퍼
# ---------------------------------------------------------------------------

def _get_audio_duration(file_path: Path) -> float:
    """ffprobe로 오디오 길이를 초 단위로 반환한다. 메모리 사용 없음."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def _detect_silence_points(file_path: Path) -> list[float]:
    """ffmpeg silencedetect로 무음 구간의 중간 지점 목록을 반환한다. 스트리밍 방식."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(file_path),
         "-af", f"silencedetect=noise={config.CHUNK_SILENCE_DB}dB:d={config.CHUNK_SILENCE_DUR}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )
    # stderr에서 silence_start/silence_end 파싱
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", result.stderr)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", result.stderr)]
    # 각 무음 구간의 중간 지점 반환
    points = []
    for s, e in zip(starts, ends):
        points.append((s + e) / 2)
    return points


def _find_split_points(
    silence_points: list[float],
    total_duration: float,
    target_chunk: int,
    margin: int,
) -> list[float]:
    """목표 청크 길이 근처의 무음 지점에서 분할 지점을 결정한다."""
    split_points = []
    current_start = 0.0

    while current_start + target_chunk < total_duration:
        target = current_start + target_chunk
        lo = target - margin
        hi = target + margin

        # 범위 내 무음 지점 중 목표에 가장 가까운 것 선택
        candidates = [p for p in silence_points if lo <= p <= hi]
        if candidates:
            best = min(candidates, key=lambda p: abs(p - target))
        else:
            best = target  # 무음 없으면 고정 분할 (폴백)

        split_points.append(best)
        current_start = best

    return split_points


def _extract_chunk(file_path: Path, start: float, end: float, output_path: Path) -> None:
    """ffmpeg로 특정 구간을 16kHz mono WAV로 추출한다."""
    duration = end - start
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(file_path),
         "-t", str(duration), "-ar", "16000", "-ac", "1",
         "-f", "wav", str(output_path)],
        capture_output=True, timeout=300,
    )


def _compute_audio_stats(
    audio: np.ndarray | None,
    sr: int,
    segments: list[dict],
    total_duration: float,
    file_size_bytes: int = 0,
) -> dict:
    """오디오 통계를 계산한다. audio가 None(청크 모드)이면 segments 기반으로 추정."""
    # segments에서 발화 시간 합산
    speech_seconds = sum(
        seg.get("end", 0) - seg.get("start", 0)
        for seg in segments
        if seg.get("end", 0) > seg.get("start", 0)
    )
    silence_ratio = max(0.0, 1.0 - speech_seconds / total_duration) if total_duration > 0 else 0.0
    effective_minutes = round(speech_seconds / 60, 2)

    stats: dict = {
        "sample_rate": sr,
        "channels": 1,
        "bitrate": round(file_size_bytes * 8 / total_duration / 1000, 1) if total_duration > 0 and file_size_bytes > 0 else 0,
        "silence_ratio": round(silence_ratio, 3),
        "effective_minutes": effective_minutes,
    }

    if audio is not None and len(audio) > 0:
        # PCM 기반 정밀 분석
        rms = float(np.sqrt(np.mean(audio ** 2)))
        stats["rms"] = round(rms, 4)

        # clipping: |sample| > 0.99 비율
        clipping_ratio = float(np.mean(np.abs(audio) > 0.99))
        stats["clipping_ratio"] = round(clipping_ratio, 5)

        # SNR 추정: 프레임별 RMS로 signal vs noise 추정
        frame_len = sr // 10  # 100ms 프레임
        n_frames = len(audio) // frame_len
        if n_frames > 1:
            frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            sorted_rms = np.sort(frame_rms)
            # 하위 10%를 노이즈, 상위 50%를 시그널로 추정
            noise_floor = float(np.mean(sorted_rms[:max(1, n_frames // 10)]))
            signal_level = float(np.mean(sorted_rms[n_frames // 2:]))
            if noise_floor > 1e-8:
                snr_db = 20 * np.log10(signal_level / noise_floor)
            else:
                snr_db = 60.0  # 노이즈가 거의 없음
            stats["snr_db"] = round(float(snr_db), 1)
        else:
            stats["snr_db"] = 0.0
    else:
        # 청크 모드: PCM 없음 → 추정 불가 필드는 null
        stats["rms"] = None
        stats["snr_db"] = None
        stats["clipping_ratio"] = None

    # qualityFactor: bitrate*0.3 + snr*0.5 + sampleRate*0.2
    bitrate_val = stats.get("bitrate", 0) or 0
    snr_val = stats.get("snr_db") if stats.get("snr_db") is not None else 0
    bitrate_score = min(1.0, bitrate_val / 192)
    snr_score = min(1.0, snr_val / 42)
    sr_score = 0.8  # 리샘플링 후 항상 16kHz mono
    stats["quality_factor"] = round(bitrate_score * 0.3 + snr_score * 0.5 + sr_score * 0.2, 2)

    return stats


def _clean_segments(raw_segments: list[dict]) -> list[dict]:
    """WhisperX 결과 세그먼트를 정리한다 (word 데이터 보존).

    raw_direct(SPEAKER_MAPPING_MODE=raw_direct) 가 부착하는 메타는 있을 때만 보존한다
    (whisperx legacy 경로에는 없으므로 추가되지 않음 → 기존 동작 무변경):
      - word: speaker_source (라벨 출처: exact/overlap/tolerance/backchannel/ambiguous)
      - segment: source_distribution / overlap_ranges / parent_segment_text
    speaker_id 산정 로직은 word.speaker 만 사용하므로(_get_speaker_id) 메타 보존은
    발화 분리/화자 배정에 영향이 없고, transcript_words JSONB 추가 키로만 흐른다.
    """
    segments = []
    for seg in raw_segments:
        segment = {
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", "").strip(),
        }
        if "speaker" in seg:
            segment["speaker"] = seg["speaker"]
        # raw_direct segment 메타 — 있을 때만 보존 (legacy whisperx 에는 부재)
        for meta_key in ("source_distribution", "overlap_ranges", "parent_segment_text"):
            if meta_key in seg:
                segment[meta_key] = seg[meta_key]
        if "words" in seg:
            segment["words"] = [
                _clean_word(w)
                for w in seg["words"]
                if w.get("start") is not None and w.get("end") is not None
            ]
        segments.append(segment)
    return segments


def _clean_word(w: dict) -> dict:
    """word dict 정리. raw_direct 의 speaker_source 메타는 있을 때만 보존한다.

    legacy whisperx word 에는 speaker_source 키가 없어 추가되지 않으므로 기존 schema
    (word/start/end/speaker) 와 호환. speaker 가 None 이어도 문자열 "None" 으로
    바꾸지 않는다(_get_speaker_id 가 None 을 그대로 받아 필터하도록 유지 — PR #22).
    """
    cleaned = {
        "word": w.get("word", ""),
        "start": round(w.get("start", 0), 2),
        "end": round(w.get("end", 0), 2),
        "speaker": w.get("speaker"),
    }
    if "speaker_source" in w:
        cleaned["speaker_source"] = w["speaker_source"]
    return cleaned


def _offset_segments(segments: list[dict], offset: float) -> list[dict]:
    """세그먼트/워드의 start/end에 오프셋을 더한다 (불변)."""
    result = []
    for seg in segments:
        new_seg = {
            **seg,
            "start": round(seg["start"] + offset, 2),
            "end": round(seg["end"] + offset, 2),
        }
        if "words" in seg and seg["words"]:
            new_seg["words"] = [
                {**w, "start": round(w["start"] + offset, 2), "end": round(w["end"] + offset, 2)}
                for w in seg["words"]
            ]
        result.append(new_seg)
    return result


def _cleanup_temp_files() -> None:
    """서버 시작 시 이전 세션의 잔여 임시 파일을 정리한다."""
    import glob
    cleaned = 0

    # TEMP_DIR: 업로드 원본 + 청크 WAV + denoise 임시 파일
    for pattern in ("*.m4a", "*.wav", "*.mp3", "*.ogg", "*.flac", "*.webm", "*.mp4", "*.raw"):
        for f in glob.glob(str(config.TEMP_DIR / pattern)):
            try:
                os.unlink(f)
                cleaned += 1
            except OSError:
                pass

    # RESULTS_DIR: 이전 세션의 WAV 결과 디렉토리
    if config.RESULTS_DIR.exists():
        import shutil
        for d in config.RESULTS_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                cleaned += 1

    if cleaned > 0:
        logger.info("이전 세션 잔여 파일 %d개 정리 완료", cleaned)


def _apply_reclustering(
    audio: "np.ndarray",
    sample_rate: int,
    result: dict,
    task_id: str,
) -> dict:
    """WeSpeaker 재클러스터링 hook (Phase 7).

    `assign_word_speakers` 직후 호출. flag/모델 미가용 시 byte-equivalent.
    WhisperX `result["segments"]`에서 word를 평탄화해 hook에 전달하고,
    반환된 새 speaker 라벨을 원본 segment의 word 위치에 다시 적용한다.
    """
    recluster_config = ReclusterConfig.from_env()
    if not recluster_config.is_enabled_for("call_recording"):
        return result

    global _speaker_embedding_model
    if _speaker_embedding_model is None:
        _speaker_embedding_model = SpeakerEmbeddingModel()

    # word를 평탄화하면서 (segment_idx, word_idx_in_segment) 위치 추적
    flat_words: list[dict] = []
    locations: list[tuple[int, int]] = []
    for s_idx, seg in enumerate(result.get("segments", [])):
        for w_idx, word in enumerate(seg.get("words", [])):
            flat_words.append(word)
            locations.append((s_idx, w_idx))

    if not flat_words:
        return result

    recluster_start = time.time()
    recluster_result = maybe_recluster_speakers(
        audio=audio,
        sample_rate=sample_rate,
        words=flat_words,
        segments=list(result.get("segments", [])),
        mode="call_recording",
        embedding_model=_speaker_embedding_model,
    )
    recluster_elapsed_ms = (time.time() - recluster_start) * 1000

    if recluster_result.changed:
        # 새 speaker_id를 WhisperX 컨벤션의 "speaker" 필드에도 동기화하면서
        # 원본 segment.words[w_idx]를 immutable copy로 교체한다.
        new_segments = [dict(seg) for seg in result["segments"]]
        for seg in new_segments:
            seg["words"] = list(seg.get("words", []))
        for new_word, (s_idx, w_idx) in zip(recluster_result.words, locations):
            updated = dict(new_word)
            new_speaker = updated.get("speaker_id")
            if new_speaker is not None:
                updated["speaker"] = new_speaker
            new_segments[s_idx]["words"][w_idx] = updated
        result["segments"] = new_segments

    logger.info(
        "[%s] WeSpeaker 재클러스터링 완료 (windows=%d, confidence=%.2f, changed=%s, %.0fms)",
        task_id,
        recluster_result.window_count,
        recluster_result.confidence,
        recluster_result.changed,
        recluster_elapsed_ms,
    )
    return result


def _do_speaker_assign(diarize_segments, result, task_id: str):
    """SPEAKER_MAPPING_MODE 분기 — raw_direct (Phase 3) | whisperx (legacy)."""
    mode = config.SPEAKER_MAPPING_MODE
    if mode == "raw_direct":
        # lazy import (whisperx 와 별개)
        from app.speaker_mapping import assign_speakers
        return assign_speakers(
            diarize_segments,
            result,
            tolerance_default_ms=config.SPEAKER_MAP_TOLERANCE_DEFAULT_MS,
            tolerance_max_ms=config.SPEAKER_MAP_TOLERANCE_MAX_MS,
            backchannel_dur_max=config.SPEAKER_MAP_BACKCHANNEL_DUR_MAX,
            overlap_min_s=config.SPEAKER_MAP_OVERLAP_MIN_SEC,
        )
    if mode == "whisperx":
        return whisperx.assign_word_speakers(diarize_segments, result)
    logger.warning(
        "[%s] unknown SPEAKER_MAPPING_MODE=%r → fallback to whisperx",
        task_id, mode,
    )
    return whisperx.assign_word_speakers(diarize_segments, result)


def _transcribe_with_oom_guard(audio, task_id: str) -> dict:
    """OOM 시 batch_size 를 절반 단계로 후퇴 (예: 4 → 2 → 1) 후 retry.

    RTX 4060 8GB 등 VRAM 압박 환경에서 large-v3 + alignment + pyannote 동시 로딩 시
    batch_size 가 메모리 피크 주범. 한 번에 OOM 나도 cache 비우고 작은 batch 로 재시도.

    config.BATCH_OOM_RETRY_ENABLED=false 면 첫 OOM 에 그대로 raise (rollback 안전망).
    """
    current = config.BATCH_SIZE
    min_bs = max(1, config.BATCH_SIZE_MIN)
    attempt = 0
    while True:
        attempt += 1
        try:
            return _model.transcribe(audio, batch_size=current)
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            logger.warning(
                "[%s] OOM at batch_size=%d (attempt=%d): %s",
                task_id, current, attempt, oom,
            )
            if not config.BATCH_OOM_RETRY_ENABLED or current <= min_bs:
                raise
            next_bs = max(min_bs, current // 2)
            if next_bs >= current:
                raise
            current = next_bs
            logger.info("[%s] OOM fallback → batch_size=%d", task_id, current)


def load_models():
    """WhisperX 모델을 전역으로 로딩한다."""
    global _model, _align_model, _align_metadata, _diarize_model

    _cleanup_temp_files()

    logger.info("WhisperX 모델 로딩 시작 (model=%s, device=%s)...", config.MODEL_SIZE, config.DEVICE)
    start = time.time()

    asr_options = {}
    if config.HOTWORDS:
        asr_options["hotwords"] = config.HOTWORDS
        logger.info("Hotwords 설정: %s", config.HOTWORDS)
    if config.INITIAL_PROMPT:
        asr_options["initial_prompt"] = config.INITIAL_PROMPT
        logger.info("Initial prompt 설정: %s", config.INITIAL_PROMPT[:50])

    _model = whisperx.load_model(
        config.MODEL_SIZE,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
        language=config.LANGUAGE,
        asr_options=asr_options if asr_options else None,
        vad_options={"vad_onset": config.VAD_ONSET, "vad_offset": config.VAD_OFFSET},
    )
    logger.info("Silero VAD 사용 (onset=%.3f, offset=%.3f)", config.VAD_ONSET, config.VAD_OFFSET)

    # Forced alignment 모델
    _align_model, _align_metadata = whisperx.load_align_model(
        language_code=config.LANGUAGE,
        device=config.DEVICE,
    )

    # 화자분리 모델 (HF_TOKEN이 있을 때만)
    if config.HF_TOKEN:
        logger.info("화자분리 모델 로딩 중 (HF_TOKEN 감지)...")
        try:
            _diarize_model = DiarizationPipeline(
                model_name=config.DIARIZATION_MODEL,
                token=config.HF_TOKEN,
                device=config.DEVICE,
            )
            logger.info("화자분리 모델 로딩 완료 (model=%s)", config.DIARIZATION_MODEL)
        except Exception as e:
            logger.warning("화자분리 모델 로딩 실패 — 화자분리 비활성화: %s", e)
            _diarize_model = None
    else:
        logger.info("HF_TOKEN 미설정 — 화자분리 비활성화")

    # DeepFilterNet: denoise 플래그가 켜진 경우에만 로딩 (VRAM/메모리 절약)
    if config.PREPROCESS_DENOISE_ENABLED:
        logger.info("DeepFilterNet 상주 워커 로딩 중 (PREPROCESS_DENOISE_ENABLED=true)")
        load_df_model()
    else:
        logger.info("PREPROCESS_DENOISE_ENABLED=false — DeepFilterNet 미로딩")

    elapsed = time.time() - start
    logger.info("모델 로딩 완료 (%.1f초)", elapsed)


def _transcribe_chunk(
    audio: np.ndarray,
    task_id: str,
    enable_diarize: bool,
    diarization_options: dict | None = None,
) -> list[dict]:
    """단일 청크를 GPU 추론하고 정리된 세그먼트를 반환한다."""
    if diarization_options is None:
        diarization_options = {}

    # GPU 추론 (전처리는 호출자가 이미 적용)
    lock_wait_start = time.time()
    _gpu_lock.acquire()
    lock_wait_ms = int((time.time() - lock_wait_start) * 1000)
    inference_start = time.time()
    job_store.update_gpu_acquired(task_id)
    logger.info("[%s] GPU lock 획득 | lock_wait_ms=%d", task_id, lock_wait_ms)
    try:
        result = _transcribe_with_oom_guard(audio, task_id)
        logger.info("[%s] Transcribe 완료 (%d 세그먼트)", task_id, len(result["segments"]))

        try:
            result = whisperx.align(
                result["segments"], _align_model, _align_metadata,
                audio, config.DEVICE, return_char_alignments=False,
            )
            logger.info("[%s] Alignment 완료", task_id)
        except Exception as align_err:
            logger.warning("[%s] Alignment 실패: %s", task_id, align_err)

        if enable_diarize and _diarize_model is not None:
            try:
                diarize_segments = _diarize_model(audio, **diarization_options)
                result = _do_speaker_assign(diarize_segments, result, task_id)
                logger.info("[%s] 화자분리 완료", task_id)

                # Phase 7: WeSpeaker reclustering (chunked path)
                result = _apply_reclustering(audio, config.SAMPLE_RATE, result, task_id)
            except Exception as diarize_err:
                logger.warning("[%s] 화자분리 실패: %s", task_id, diarize_err)
    finally:
        inference_ms = int((time.time() - inference_start) * 1000)
        torch.cuda.empty_cache()
        _gpu_lock.release()
        job_store.update_gpu_released(task_id)
        logger.info(
            "[%s] GPU lock 해제 | inference_ms=%d lock_wait_ms=%d (VRAM 정리 완료)",
            task_id, inference_ms, lock_wait_ms,
        )

    return _clean_segments(result["segments"])


def _transcribe_chunked(
    file_path: Path,
    task_id: str,
    total_duration: float,
    enable_diarize: bool,
    split_by_utterance: bool = False,
    diarization_options: dict | None = None,
    mask_audio_pii: bool = False,
    mask_audio_names: bool = False,
    pii_intervals_only: bool = False,
) -> tuple[list[dict], list[dict], dict[str, bytes], list[tuple[float, float, str]]]:
    """대용량 오디오를 무음 기반으로 청크 분할하여 처리한다.

    Returns:
        (all_segments, all_utterances, audio_files, pii_audio_ranges)

    `split_by_utterance=True`이고 화자분리가 활성화된 경우, 각 청크의
    `chunk_audio`가 메모리에 상주해 있는 동안 chunk-local 좌표계로
    utterance 경계를 산출하고 바로 WAV 바이트를 생성한다. 이후 메타데이터
    타임스탬프를 누적 offset으로 globalize한다. 전체 원본을 한 번도 메모리에
    올리지 않으므로 청크 모드의 OOM 제약을 그대로 유지한다.
    """
    if diarization_options is None:
        diarization_options = {}

    logger.info("[%s] 청크 모드 시작 (총 %.0f초, 목표 청크 %ds)", task_id, total_duration, config.CHUNK_DURATION_SEC)

    # 1. 무음 지점 탐지
    silence_points = _detect_silence_points(file_path)
    logger.info("[%s] 무음 지점 %d개 감지", task_id, len(silence_points))

    # 2. 분할 지점 결정
    split_points = _find_split_points(
        silence_points, total_duration,
        config.CHUNK_DURATION_SEC, config.CHUNK_MARGIN_SEC,
    )
    logger.info("[%s] 분할 지점 %d개: %s", task_id, len(split_points),
                [f"{p:.0f}s" for p in split_points])

    # 3. 청크 경계 계산
    boundaries = [0.0] + split_points + [total_duration]
    all_segments: list[dict] = []
    all_utterances: list[dict] = []
    audio_files: dict[str, bytes] = {}
    all_pii_audio_ranges: list[tuple[float, float, str]] = []
    global_utt_idx = 0
    # 전처리된 연속 타임라인 누적 offset — 각 청크의 전처리 후 실제 길이를 누적하여
    # 청크 경계에서 silence compression 등으로 삭제된 구간이 타임스탬프 gap으로
    # 나타나지 않도록 보정한다.
    cumulative_preprocessed_offset = 0.0
    diarize_active = enable_diarize and _diarize_model is not None
    emit_utterances = split_by_utterance and diarize_active

    for i in range(len(boundaries) - 1):
        chunk_start = boundaries[i]
        chunk_end = boundaries[i + 1]
        chunk_idx = i + 1
        total_chunks = len(boundaries) - 1

        logger.info("[%s] 청크 %d/%d 처리 중 (%.0fs~%.0fs, %.0f초)",
                    task_id, chunk_idx, total_chunks, chunk_start, chunk_end, chunk_end - chunk_start)

        # ffmpeg로 청크 WAV 추출
        chunk_path = config.TEMP_DIR / f"{task_id}_chunk_{i:03d}.wav"
        _extract_chunk(file_path, chunk_start, chunk_end, chunk_path)

        try:
            # 청크 로딩 + 전처리 + 처리
            chunk_audio = whisperx.load_audio(str(chunk_path))
            original_chunk_duration = len(chunk_audio) / config.SAMPLE_RATE
            chunk_audio = preprocess(chunk_audio, config.SAMPLE_RATE)
            preprocessed_chunk_duration = len(chunk_audio) / config.SAMPLE_RATE

            if preprocessed_chunk_duration < original_chunk_duration - 0.1:
                logger.info(
                    "[%s] 청크 %d/%d 전처리 길이 변경: %.1fs → %.1fs (%.1fs 감소)",
                    task_id, chunk_idx, total_chunks,
                    original_chunk_duration, preprocessed_chunk_duration,
                    original_chunk_duration - preprocessed_chunk_duration,
                )

            chunk_segments = _transcribe_chunk(chunk_audio, task_id, enable_diarize, diarization_options)

            # 4. 음성 PII 마스킹 (청크 모드)
            # D4b: range 산출 게이트 = (mask_audio_pii OR pii_intervals_only),
            # 오디오 변형(beep) 게이트 = mask_audio_pii 단독. pii_intervals_only 는
            # chunk_audio 를 변형하지 않으므로 emit 되는 발화 WAV 가 원본과 동일하게 유지된다.
            if mask_audio_pii or pii_intervals_only:
                chunk_pii_ranges = find_pii_word_ranges(
                    chunk_segments,
                    enable_name_masking=mask_audio_names,
                    pad_sec=config.PII_MASK_PAD_SEC,
                )
                if chunk_pii_ranges:
                    if mask_audio_pii:
                        chunk_audio = mask_audio_ranges(chunk_audio, chunk_pii_ranges, config.SAMPLE_RATE)
                    # 글로벌 타임라인으로 변환하여 저장 (beep 여부와 무관하게 range 기록)
                    for s, e, t in chunk_pii_ranges:
                        all_pii_audio_ranges.append((
                            s + cumulative_preprocessed_offset,
                            e + cumulative_preprocessed_offset,
                            t
                        ))

            # 청크 내 발화 분리 + WAV 생성 (chunk_audio가 살아있는 동안 수행)
            if emit_utterances:
                chunk_utts, chunk_files, global_utt_idx = emit_chunk_utterances(
                    chunk_audio,
                    chunk_segments,
                    preprocessed_chunk_duration,
                    cumulative_preprocessed_offset,
                    global_utt_idx,
                    config.SAMPLE_RATE,
                )
                all_utterances.extend(chunk_utts)
                audio_files.update(chunk_files)

            # 타임스탬프를 전처리된 연속 타임라인에 배치 (누적 offset 사용)
            offset_segments = _offset_segments(chunk_segments, cumulative_preprocessed_offset)
            all_segments.extend(offset_segments)

            cumulative_preprocessed_offset += preprocessed_chunk_duration

            logger.info("[%s] 청크 %d/%d 완료 (%d 세그먼트, 누적 %.1fs)",
                        task_id, chunk_idx, total_chunks, len(chunk_segments), cumulative_preprocessed_offset)
        finally:
            # 청크 파일 삭제 + 메모리 해제
            chunk_path.unlink(missing_ok=True)
            try:
                del chunk_audio
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()

    logger.info(
        "[%s] 청크 모드 완료 (총 %d 세그먼트, %d 발화, %d개 PII 마스킹, 전처리 후 총 길이 %.1fs / 원본 %.1fs)",
        task_id, len(all_segments), len(all_utterances), len(all_pii_audio_ranges),
        cumulative_preprocessed_offset, total_duration,
    )
    return all_segments, all_utterances, audio_files, all_pii_audio_ranges


def transcribe(
    file_path: str,
    task_id: str,
    enable_diarize: bool = False,
    enable_name_masking: bool = False,
    mask_pii: bool = True,
    split_by_speaker: bool = False,
    split_by_utterance: bool = False,
    denoise_enabled: bool | None = None,
    mask_audio_pii: bool = False,
    mask_audio_names: bool = False,
    pii_intervals_only: bool = False,
    reference_embedding: list[float] | None = None,
) -> dict:
    """음성 파일을 STT 처리하고 마스킹된 결과를 반환한다.

    오디오가 CHUNK_THRESHOLD_SEC 이상이면 무음 기반 청크 분할 모드로 전환하여
    메모리 사용량을 일정하게 유지한다.
    """
    file_path = Path(file_path)

    try:
        logger.info("[%s] STT 시작: %s", task_id, file_path.name)
        start = time.time()

        # Phase 2 (Option D): Load diarization config once at entry.
        # Default mode to "call_recording" for all calls (voice-api has no mode concept yet).
        diarization_config = DiarizationConfig.from_env()
        diarization_options = diarization_config.resolve_options("call_recording")

        # 0. 오디오 길이 확인 (메모리 사용 없음)
        total_duration = _get_audio_duration(file_path)
        use_chunked = total_duration > config.CHUNK_THRESHOLD_SEC

        # 청크 모드에서만 채워지는 버킷. 일반 모드 경로에서도 이름이 정의돼 있도록
        # 미리 초기화해 short-circuit 조건에 의존하지 않게 한다.
        chunked_utterances: list[dict] = []
        chunked_audio_files: dict[str, bytes] = {}
        pii_audio_ranges: list[tuple[float, float, str]] = []
        pre_mask_texts_by_speaker: dict[str, list[str]] = {}
        speakers_result: list[dict] | None = None

        if use_chunked:
            # ── 청크 모드: 대용량 오디오 ──
            # 발화 WAV는 청크 내부에서 바로 생성된다. 화자별 WAV는 전체 배열이 필요하므로
            # 청크 모드에서는 제공하지 않는다 (API 스펙에 명시).
            # PII 이름 마스킹: 텍스트(enable_name_masking)와 음성(mask_audio_names)을 OR로
            # 동기화. 사용자가 텍스트만 켜도 음성 PII도 자동 마스킹되어 정합성 유지.
            audio_name_masking = mask_audio_names or enable_name_masking
            segments, chunked_utterances, chunked_audio_files, pii_audio_ranges = _transcribe_chunked(
                file_path, task_id, total_duration, enable_diarize,
                split_by_utterance=split_by_utterance,
                diarization_options=diarization_options,
                mask_audio_pii=mask_audio_pii,
                mask_audio_names=audio_name_masking,
                pii_intervals_only=pii_intervals_only,
            )
            audio = None
            diarize_active = enable_diarize and _diarize_model is not None
        else:
            # ── 일반 모드: 전체 로딩 + 전처리 ──
            audio = whisperx.load_audio(str(file_path))
            logger.info("[%s] 오디오 로드 완료 (%.1fs)", task_id, len(audio) / config.SAMPLE_RATE)
            audio = preprocess(audio, config.SAMPLE_RATE)

            lock_wait_start = time.time()
            _gpu_lock.acquire()
            lock_wait_ms = int((time.time() - lock_wait_start) * 1000)
            inference_start = time.time()
            job_store.update_gpu_acquired(task_id)
            logger.info("[%s] GPU lock 획득 | lock_wait_ms=%d", task_id, lock_wait_ms)
            try:
                result = _transcribe_with_oom_guard(audio, task_id)
                logger.info("[%s] Transcribe 완료 (%d 세그먼트)", task_id, len(result["segments"]))

                try:
                    result = whisperx.align(
                        result["segments"], _align_model, _align_metadata,
                        audio, config.DEVICE, return_char_alignments=False,
                    )
                    logger.info("[%s] Alignment 완료", task_id)
                except Exception as align_err:
                    logger.warning("[%s] Alignment 실패: %s", task_id, align_err)

                try:
                    if enable_diarize and _diarize_model is not None:
                        diarize_segments = _diarize_model(audio, **diarization_options)
                        result = _do_speaker_assign(diarize_segments, result, task_id)
                        logger.info("[%s] 화자분리 완료", task_id)

                        # Phase 7: WeSpeaker reclustering (non-chunked path)
                        result = _apply_reclustering(audio, config.SAMPLE_RATE, result, task_id)
                    elif enable_diarize and _diarize_model is None:
                        logger.warning("[%s] 화자분리 요청했으나 HF_TOKEN 미설정으로 건너뜀", task_id)
                except Exception as diarize_err:
                    logger.warning("[%s] 화자분리 실패: %s", task_id, diarize_err)
            finally:
                inference_ms = int((time.time() - inference_start) * 1000)
                torch.cuda.empty_cache()
                _gpu_lock.release()
                job_store.update_gpu_released(task_id)
                logger.info(
                    "[%s] GPU lock 해제 | inference_ms=%d lock_wait_ms=%d (VRAM 정리 완료)",
                    task_id, inference_ms, lock_wait_ms,
                )

            segments = _clean_segments(result["segments"])
            diarize_active = enable_diarize and _diarize_model is not None

            # 5. 음성 PII 마스킹 (일반 모드)
            # PII 이름 마스킹: 텍스트와 음성을 OR로 동기화 (chunked 모드와 동일 정책).
            audio_name_masking = mask_audio_names or enable_name_masking
            # D4b: range 산출 게이트 = (mask_audio_pii OR pii_intervals_only).
            # 오디오 변형(1kHz beep) 게이트 = mask_audio_pii 단독.
            # pii_intervals_only 는 time_range 메타데이터만 산출하고 audio 를 변형하지 않으므로
            # 이 경로에서 잘려나가는 발화 WAV 는 비-PII 경로와 바이트 동일하게 유지된다.
            if mask_audio_pii or pii_intervals_only:
                pii_audio_ranges = find_pii_word_ranges(
                    segments,
                    enable_name_masking=audio_name_masking,
                    pad_sec=config.PII_MASK_PAD_SEC,
                )
                if pii_audio_ranges and mask_audio_pii:
                    audio = mask_audio_ranges(audio, pii_audio_ranges, config.SAMPLE_RATE)
                    logger.info("[%s] 음성 PII 마스킹 완료 (%d개 구간)", task_id, len(pii_audio_ranges))

        # STAGE 15: PII 마스킹 전 화자별 텍스트 스냅샷 (호칭어 기반 관계 탐지용)
        # mask_segments()가 segments 텍스트를 in-place 치환하므로 그 전에 수집해야 한다.
        if enable_diarize and diarize_active:
            for seg in segments:
                spk = seg.get("speaker") or "SPEAKER_00"
                pre_mask_texts_by_speaker.setdefault(spk, []).append(seg.get("text", ""))

        # 5. PII 마스킹 (텍스트)
        pii_summary = mask_segments(segments, enable_name_masking) if mask_pii else []

        # 5.5 pii_summary에 음성 마스킹 시간 범위 통합 (immutable)
        # D4b: pii_intervals_only 모드에서도 time_ranges 는 emit 한다 (beep 없이 메타데이터만).
        if (mask_audio_pii or pii_intervals_only) and pii_audio_ranges:
            type_to_ranges: dict[str, list[dict]] = {}
            for r_start, r_end, p_type in pii_audio_ranges:
                type_to_ranges.setdefault(p_type, []).append(
                    {"start": round(r_start, 2), "end": round(r_end, 2)}
                )
            existing_types = {item["type"] for item in pii_summary}
            pii_summary = [
                {**item, "time_ranges": type_to_ranges[item["type"]]}
                if item["type"] in type_to_ranges
                else {**item}
                for item in pii_summary
            ]
            # PR-B2: type_to_ranges 에 있으나 mask_segments 의 pii_summary 에 없는 type
            # (PR-B extended detector 의 credential_like / foreign_id_like / payment_like /
            # numeric_sensitive_like / korean_name_like_candidate) 에 대해 신규 항목 추가.
            # mask_pii 의 pattern_order 는 PII_PATTERNS + "이름" 만 emit 하므로 extended
            # type 은 pii_summary 에 미존재 → 본 단계가 없으면 find_pii_word_ranges 가
            # 시간범위를 산출해도 build_pii_intervals 입력으로 흐르지 않는다 (PR-B2 단절점).
            # count = time_ranges 길이 (PIIDetectedItem.count >= 1 보장). type 자체에
            # `_candidate` suffix 가 후보 표지 (worker.build_pii_intervals 무변경, 호출자가
            # pii_extended.is_candidate_type(piiType) 으로 판별). D4b text_only 정책 보존
            # (mask_type 은 worker.build_pii_intervals 의 PII_INTERVAL_MASK_TYPE 그대로).
            for p_type, ranges in type_to_ranges.items():
                if p_type not in existing_types:
                    pii_summary.append({
                        "type": p_type,
                        "count": len(ranges),
                        "time_ranges": ranges,
                    })

        # 6. 전체 텍스트
        full_text = " ".join(s["text"] for s in segments)

        # 6.5 화자/발화 분리 (일반 모드 + diarize 활성화 시에만)
        utterances_result = None
        speaker_audio_result = None
        audio_files = {}

        # 청크 모드에서는 _transcribe_chunked가 발화/WAV를 이미 생성했다.
        # 화자별 WAV(speaker_audio)는 청크 모드에서 제공되지 않는다.
        if use_chunked and split_by_utterance and diarize_active and chunked_utterances:
            utterances_result = chunked_utterances
            audio_files.update(chunked_audio_files)
            logger.info("[%s] 청크 모드 발화 %d개 복원", task_id, len(utterances_result))

        if audio is not None and diarize_active and (split_by_speaker or split_by_utterance):
            total_dur = len(audio) / config.SAMPLE_RATE

            if split_by_utterance:
                all_words = []
                for s in segments:
                    if s.get("words"):
                        for w in s["words"]:
                            flat = {
                                "word": w.get("word", ""),
                                "start": w.get("start", s["start"]),
                                "end": w.get("end", s["end"]),
                                "speaker": w.get("speaker", s.get("speaker")),
                            }
                            # raw_direct word 메타 보존 (있을 때만; legacy 에는 부재)
                            if "speaker_source" in w:
                                flat["speaker_source"] = w["speaker_source"]
                            all_words.append(flat)
                    else:
                        all_words.append({
                            "word": s.get("text", ""),
                            "start": s["start"],
                            "end": s["end"],
                            "speaker": s.get("speaker"),
                        })

                for i, w in enumerate(all_words):
                    if w["speaker"] is None:
                        if i > 0 and all_words[i - 1]["speaker"] is not None:
                            w["speaker"] = all_words[i - 1]["speaker"]
                        elif i + 1 < len(all_words) and all_words[i + 1]["speaker"] is not None:
                            w["speaker"] = all_words[i + 1]["speaker"]
                        else:
                            # SPEAKER_NN 패턴 (자릿수 패딩) 유지 — 다른 코드 경로와 일관성
                            w["speaker"] = "SPEAKER_00"

                utterance_boundaries = segment_utterances(all_words, total_dur)
                utterances_result = []
                for idx, utt in enumerate(utterance_boundaries):
                    utt_audio = extract_utterance_audio(audio, utt, config.SAMPLE_RATE)
                    filename = f"utterance_{idx:03d}.wav"
                    audio_files[filename] = to_wav_bytes(utt_audio, config.SAMPLE_RATE)
                    utterances_result.append({
                        "index": idx,
                        "start_sec": utt.start_sec,
                        "end_sec": utt.end_sec,
                        "duration_sec": utt.duration_sec,
                        "speaker_id": utt.speaker_id,
                        "transcript_text": utt.transcript_text,
                        "audio_filename": filename,
                        "words": list(utt.words),
                    })
                logger.info("[%s] 발화 %d개 분리 완료", task_id, len(utterances_result))

            if split_by_speaker:
                speaker_ids = sorted(set(
                    s.get("speaker", "SPEAKER_00") for s in segments
                    if s.get("speaker") is not None
                ))
                speaker_audio_result = []
                for sid in speaker_ids:
                    muted = mute_non_speaker(audio, segments, sid, config.SAMPLE_RATE)
                    filename = f"speaker_{sid.lower()}.wav"
                    audio_files[filename] = to_wav_bytes(muted, config.SAMPLE_RATE)
                    speaker_audio_result.append({
                        "speaker_id": sid,
                        "total_duration_sec": round(len(audio) / config.SAMPLE_RATE, 2),
                        "audio_filename": filename,
                    })
                logger.info("[%s] 화자 %d명 오디오 분리 완료", task_id, len(speaker_audio_result))

        # 전처리(+ PII 마스킹) 완료된 오디오를 _preprocessed_audio.wav 로 출력.
        # gpu-worker 가 이 파일을 S3 의 raw_audio_url 경로에 덮어써서
        # 발화 startSec/endSec 가 재생 위치와 일치하도록 한다.
        # 청크 모드(audio is None)에서는 단일 배열이 없으므로 생략.
        if audio is not None:
            audio_files["_preprocessed_audio.wav"] = to_wav_bytes(audio, config.SAMPLE_RATE)
            logger.info("[%s] _preprocessed_audio.wav 추가됨 (%.1fs)", task_id, len(audio) / config.SAMPLE_RATE)

        elapsed = time.time() - start

        # Bug 7 안전망: pyannote/recluster 결과의 speaker_id 갭(SPEAKER_03 누락 등)이나
        # 비표준 형식("SPEAKER_0")을 0~N-1 연속 번호로 정규화.
        # segments / utterances / speaker_audio 모두 일관되게 갱신.
        renumber_speakers_in_place(
            segments=segments,
            utterances=utterances_result,
            speaker_audio=speaker_audio_result,
        )

        # STAGE 15: 화자 자동 식별 + 속성 분석
        if enable_diarize and diarize_active and not use_chunked:
            try:
                from app.services.speaker_analysis_service import analyze_speakers

                speaker_results = analyze_speakers(
                    audio=audio,
                    sample_rate=config.SAMPLE_RATE,
                    segments=segments,
                    pre_mask_texts_by_speaker=pre_mask_texts_by_speaker,
                    reference_embedding=reference_embedding,
                    embedding_model=_speaker_embedding_model,
                )
                speakers_result = [
                    {
                        "speaker_label": r.speaker_label,
                        "speaker_role": r.speaker_role,
                        "speaker_role_source": r.speaker_role_source,
                        "speaker_gender": r.speaker_gender,
                        "speaker_voice_age_range": r.speaker_voice_age_range,
                        "speaker_speech_age_range": r.speaker_speech_age_range,
                        "speaker_speech_age_model_version": r.speaker_speech_age_model_version,
                        "speaker_relation": r.speaker_relation,
                    }
                    for r in speaker_results.values()
                ]
                logger.info("[%s] 화자 분석 완료 (%d명)", task_id, len(speakers_result))
            except Exception as spk_err:
                logger.warning("[%s] 화자 분석 실패 (graceful degradation): %s", task_id, spk_err)

        # 자동 감정/대화행위 라벨링 (모델 없을 때 graceful degradation)
        from app.services.auto_label_service import auto_label_service
        if utterances_result and auto_label_service.is_available():
            texts = [u["transcript_text"] for u in utterances_result]
            labels = auto_label_service.predict(texts)
            for u, lbl in zip(utterances_result, labels):
                u["emotion"] = lbl.emotion
                u["emotion_confidence"] = lbl.emotion_confidence
                u["dialog_act"] = lbl.dialog_act
                u["dialog_act_confidence"] = lbl.dialog_act_confidence
                u["auto_label_model_version"] = lbl.model_version

        # STAGE 16: 주제 세그먼트 탐지 (발화가 3개 이상일 때만 의미 있음)
        topic_segments_result: list[dict] | None = None
        if utterances_result and len(utterances_result) >= 3:
            try:
                from app.services.topic_segmentation_service import segment_topics

                topic_segs = segment_topics(utterances_result)
                topic_segments_result = [
                    {
                        "segment_index": s.segment_index,
                        "topic": s.topic,
                        "start_ms": s.start_ms,
                        "end_ms": s.end_ms,
                        "utterance_indices": s.utterance_indices,
                    }
                    for s in topic_segs
                ]
                logger.info("[%s] 주제 세그먼트 %d개 생성", task_id, len(topic_segments_result))
            except Exception as topic_err:
                logger.warning("[%s] 주제 세그먼트 분석 실패 (graceful degradation): %s", task_id, topic_err)

        # 오디오 통계 계산
        file_size = file_path.stat().st_size if file_path.exists() else 0
        audio_stats = _compute_audio_stats(
            audio, config.SAMPLE_RATE, segments, total_duration, file_size,
        )

        # ffprobe metadata가 손상된 컨테이너에서 잘못된 값을 반환하는 케이스를 방어한다.
        # 실제 처리된 오디오 길이를 다음 우선순위로 결정:
        #   1) audio 배열이 있으면 len(audio) / sample_rate (정확)
        #   2) segments의 max(end) (Whisper 처리 범위)
        #   3) ffprobe 메타데이터 (fallback)
        max_segment_end = max(
            (float(seg.get("end", 0.0)) for seg in segments),
            default=0.0,
        )
        if audio is not None:
            audio_duration = float(len(audio)) / float(config.SAMPLE_RATE)
            actual_duration = max(audio_duration, max_segment_end)
        else:
            # 청크 모드: audio 배열이 None일 수 있으므로 segments + total_duration 중 큰 값 사용
            actual_duration = max(max_segment_end, float(total_duration))

        output = {
            "task_id": task_id,
            "status": "completed",
            "language": config.LANGUAGE,
            "duration_seconds": round(actual_duration, 2),
            "processing_seconds": round(elapsed, 2),
            "segments": segments,
            "full_text": full_text,
            "pii_summary": pii_summary,
            "diarization_enabled": diarize_active,
            "audio_stats": audio_stats,
        }

        output["schema_version"] = 2

        if utterances_result is not None:
            output["utterances"] = utterances_result
        if speaker_audio_result is not None:
            output["speaker_audio"] = speaker_audio_result
        if speakers_result is not None:
            output["speakers"] = speakers_result
        if topic_segments_result is not None:
            output["topic_segments"] = topic_segments_result
        if audio_files:
            output["_audio_files"] = audio_files
        if use_chunked:
            output["chunked_processing"] = True

        logger.info("[%s] 완료 (%.1f초, PII %d건 마스킹%s)",
                    task_id, elapsed, len(pii_summary),
                    ", 청크 모드" if use_chunked else "")
        return output

    finally:
        # 원본 음성 파일 삭제
        try:
            if file_path.exists():
                os.unlink(file_path)
                logger.info("[%s] 음성 파일 삭제 완료", task_id)
        except OSError as e:
            logger.warning("[%s] 음성 파일 삭제 실패: %s", task_id, e)

        # 청크 모드 잔여 파일 정리 (OOM/크래시 대비)
        import glob
        for chunk_file in glob.glob(str(config.TEMP_DIR / f"{task_id}_chunk_*.wav")):
            try:
                os.unlink(chunk_file)
            except OSError:
                pass
