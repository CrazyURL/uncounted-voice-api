"""
GPU Worker — Python port of gpu-worker.ts
Polls Supabase for pending audio sessions and processes them via the voice API.
Run as a standalone process on the GPU server alongside uncounted-voice-api.
"""

import asyncio
import functools
import logging
import os
import signal
import subprocess
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
import boto3
from botocore.config import Config
from supabase import create_client, Client

from app.sanitize_json import sanitize_json_safe
from app.worker_config import resolve_voice_api_max_wait_sec

# TODO(post-seed): worker_heartbeats 테이블 + 어드민 5분 무응답 알람

# ── Constants (matched 1:1 with gpu-worker.ts) ────────────────────────
POLL_INTERVAL_SEC = 30
POLL_BACKOFF_503_SEC = 60           # 503 → 60s backoff, NO retry_count increment
STUCK_SWEEP_INTERVAL_SEC = 300      # 5 min
STUCK_THRESHOLD_SEC = 600           # 10 min
RETRY_DELAY_SEC = 1800              # 30 min between retries
MAX_RETRY_COUNT = 3
VOICE_API_POLL_INTERVAL_SEC = 1
# 장통화(>5분) 처리 시 voice-api 응답을 더 기다릴 수 있도록 env 로 조정 가능.
# 미설정=300(기존 동작 보존), invalid=ValueError 즉시 fail-loud(WORKER_CONCURRENCY 컨벤션).
# 모듈 로드 시점에 한 번만 읽음(live reload 의도 없음). 상세=app/worker_config.py docstring.
VOICE_API_MAX_WAIT_SEC = resolve_voice_api_max_wait_sec()

# D4b: worker 는 voice-api 에 pii_intervals_only 만 요청한다(시간범위 메타데이터만 산출,
# 1kHz 비프 미적용 → 저장 발화 WAV 원본 유지). mask_audio_pii(D5 음향 마스킹)는 요청하지 않음.
# 저장 WAV 가 마스킹되지 않았으므로 persist 되는 pii_intervals 의 maskType 은 "text_only"
# (텍스트만 마스킹, 오디오 원본, 마스킹은 admin 스트리밍/export 소비 시점에 적용). 비프 경로면 "audio".
PII_INTERVAL_MASK_TYPE = "text_only"

WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))
VOICE_API_URL = os.getenv("VOICE_API_URL", "http://localhost:8001")

# ── Required environment ──────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
S3_ENDPOINT_URL = os.environ["S3_ENDPOINT_URL"]
S3_AUDIO_BUCKET = os.getenv("S3_AUDIO_BUCKET", "uncounted-audio")
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

log = logging.getLogger("gpu_worker")
is_shutting_down = False

_supabase: Optional[Client] = None
_s3 = None
_http: Optional[aiohttp.ClientSession] = None


class Voice503Error(Exception):
    """voice_api 503 — reset to pending, backoff 60s, no retry_count bump."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run(fn, *args, **kwargs):
    """Run a blocking call in the default thread pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# ── Session selection ─────────────────────────────────────────────────

async def _try_claim(session_id: str, from_status: str) -> bool:
    """
    Conditional UPDATE — only succeeds if gpu_upload_status still equals from_status.
    Returns False if another worker already claimed the session.
    """
    result = await _run(
        lambda: _supabase.table("sessions").update({
            "gpu_upload_status": "running",
            "gpu_started_at": _now_iso(),
        }).eq("id", session_id).eq("gpu_upload_status", from_status).execute()
    )
    return bool(result.data)


async def pick_next_session() -> Optional[dict]:
    """
    Pick one session and atomically claim it.
    Priority: pending → failed (retry_count < 3, 30min elapsed).
    Returns None if nothing available or lost every race.
    """
    # Pending sessions (no raw_audio_url guard — DB schema ensures it's set before 'pending')
    result = await _run(
        lambda: _supabase.table("sessions").select("*")
        .eq("gpu_upload_status", "pending")
        .filter("raw_audio_url", "not.is", "null")
        .order("updated_at")
        .limit(5)
        .execute()
    )
    candidates = result.data or []

    if not candidates:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=RETRY_DELAY_SEC)).isoformat()
        result = await _run(
            lambda: _supabase.table("sessions").select("*")
            .eq("gpu_upload_status", "failed")
            .lt("gpu_retry_count", MAX_RETRY_COUNT)
            .lt("updated_at", cutoff)
            .order("updated_at")
            .limit(5)
            .execute()
        )
        candidates = result.data or []

    for session in candidates:
        if await _try_claim(session["id"], session["gpu_upload_status"]):
            return session

    return None


# ── S3 audio download ─────────────────────────────────────────────────

async def download_raw_audio(raw_audio_url: str) -> str:
    """Stream audio to a temp file on disk (never in RAM). Returns path — caller deletes.

    Handles two formats:
    - Full HTTP/HTTPS URL (old sessions: Supabase storage URL or presigned URL)
    - Bare S3 key (new sessions: object key relative to S3_AUDIO_BUCKET)
    """
    # Strip query string before extracting extension (presigned URLs: ?X-Amz-...)
    path_part = raw_audio_url.split("?")[0]
    ext = path_part.rsplit(".", 1)[-1].lower() if "." in path_part else "wav"
    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        if raw_audio_url.startswith(("http://", "https://")):
            # Legacy sessions store a full URL — download via HTTP
            async with _http.get(raw_audio_url) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    tmp.write(chunk)
        else:
            # New sessions store a bare S3 key
            await _run(lambda: _s3.download_fileobj(S3_AUDIO_BUCKET, raw_audio_url, tmp))
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


# ── Voice API ─────────────────────────────────────────────────────────

def build_submit_params() -> dict:
    """voice-api /transcribe 요청 query params. D4b: pii_intervals_only 만 추가하고
    mask_audio_pii(D5 비프)는 전달하지 않는다(voice-api 기본 false 유지)."""
    return {
        "language": "ko",
        "diarize": "true",
        "split_by_utterance": "true",
        "split_by_speaker": "true",
        "mask_pii": "true",
        "denoise": "true",
        # PII 시간범위 메타데이터만 산출(오디오 미변형). 저장 발화 WAV 원본 유지.
        "pii_intervals_only": "true",
    }


async def submit_to_voice_api(audio_path: str) -> str:
    """POST audio file to voice_api; return task_id. 503 → Voice503Error."""
    params = build_submit_params()
    url = f"{VOICE_API_URL}/api/v1/transcribe"
    ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else "wav"
    with open(audio_path, "rb") as f:
        form = aiohttp.FormData()
        form.add_field("file", f, filename=f"raw.{ext}", content_type="application/octet-stream")
        async with _http.post(url, params=params, data=form) as resp:
            if resp.status == 503:
                text = await resp.text()
                raise Voice503Error(f"voice_api 503: {text[:500]}")
            resp.raise_for_status()
            body = await resp.json()
            return body["task_id"]


async def poll_job(task_id: str) -> dict:
    """Poll job status until completed/failed; 5-min max at 1-s interval."""
    url = f"{VOICE_API_URL}/api/v1/jobs/{task_id}"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + VOICE_API_MAX_WAIT_SEC

    while True:
        async with _http.get(url) as resp:
            if resp.status >= 500:
                raise Exception(f"poll_job {task_id}: HTTP {resp.status}")
            resp.raise_for_status()
            body = await resp.json()

        status = body.get("status")
        if status == "completed":
            return body
        if status == "failed":
            raise Exception(f"voice_api job failed: {body.get('error', '')[:500]}")

        if loop.time() >= deadline:
            raise Exception(f"poll_job timeout {VOICE_API_MAX_WAIT_SEC}s: {task_id}")

        await asyncio.sleep(VOICE_API_POLL_INTERVAL_SEC)


async def download_utterance_wav(task_id: str, filename: str, dest_path: str) -> None:
    """Stream utterance WAV from voice_api to dest_path."""
    url = f"{VOICE_API_URL}/api/v1/jobs/{task_id}/audio/{filename}"
    async with _http.get(url) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(65536):
                f.write(chunk)


# ── Quality metrics (ported from qualityMetricsService.ts + ffmpegProcessor.ts) ──

def _get_audio_stats_sync(wav_path: str) -> dict:
    """Blocking: run ffmpeg to extract RMS/Peak dB and silence ratio."""
    # RMS and Peak levels via astats filter
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", wav_path,
         "-af", "astats=metadata=1:reset=0",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.warning("ffmpeg astats failed (rc=%d) for %s: %s", proc.returncode, wav_path, proc.stderr[:200])
    rms_db = -60.0
    peak_db = -60.0
    for line in proc.stderr.splitlines():
        if "RMS level dB" in line:
            try:
                rms_db = float(line.rsplit(":", 1)[-1].strip())
            except ValueError:
                pass
        elif "Peak level dB" in line:
            try:
                peak_db = float(line.rsplit(":", 1)[-1].strip())
            except ValueError:
                pass

    # Duration for silence ratio denominator
    dur_proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
        capture_output=True, text=True,
    )
    try:
        duration_sec = float(dur_proc.stdout.strip())
    except ValueError:
        duration_sec = 1.0

    # Silence detection
    sil_proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", wav_path,
         "-af", "silencedetect=noise=-40dB:d=0.3",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if sil_proc.returncode != 0:
        log.warning("ffmpeg silencedetect failed (rc=%d) for %s: %s", sil_proc.returncode, wav_path, sil_proc.stderr[:200])
    total_silence = 0.0
    sil_start = None
    for line in sil_proc.stderr.splitlines():
        if "silence_start" in line:
            try:
                sil_start = float(line.split("silence_start:")[-1].strip())
            except ValueError:
                pass
        elif "silence_end" in line and sil_start is not None:
            try:
                sil_end = float(line.split("silence_end:")[-1].split("|")[0].strip())
                total_silence += max(0.0, sil_end - sil_start)
                sil_start = None
            except ValueError:
                pass

    silence_ratio = min(1.0, total_silence / max(duration_sec, 0.001))
    return {"rms_db": rms_db, "peak_db": peak_db, "silence_ratio": silence_ratio}


def _compute_quality(rms_db: float, peak_db: float, silence_ratio: float) -> tuple:
    """Port of qualityMetricsService.ts computeQualityScore. Returns (score, grade, snr_db, speech_ratio)."""
    snr_db = abs(peak_db - rms_db)
    speech_ratio = max(0.0, 1.0 - silence_ratio)
    clipping_ratio = min(1.0, (peak_db + 1) / 1.0) if peak_db > -1 else 0.0
    snr_score = min(100.0, max(0.0, snr_db * 3))        # 33 dB → 100
    speech_score = min(100.0, speech_ratio * 120)         # 83 %+ → 100
    clipping_penalty = clipping_ratio * 30                # up to -30
    score = round(max(0.0, min(100.0,
        snr_score * 0.4 + speech_score * 0.4 + 20 - clipping_penalty
    )))
    grade = "A" if score >= 80 else "B" if score >= 50 else "C"
    return score, grade, snr_db, speech_ratio


# ── Persist results ───────────────────────────────────────────────────

def build_pii_intervals(
    pii_summary: list[dict],
    utt_start: float,
    utt_end: float,
    mask_type: str = PII_INTERVAL_MASK_TYPE,
) -> list[dict]:
    """job-level pii_summary.time_ranges 중 발화 구간 [utt_start, utt_end) 과 겹치는 것만
    utterance-level pii_intervals 로 변환한다. maskType=저장 WAV provenance(D4b: text_only).
    원문 텍스트는 포함하지 않는다(startSec/endSec/maskType/piiType 만)."""
    intervals: list[dict] = []
    for pii_item in pii_summary:
        for tr in pii_item.get("time_ranges", []):
            if tr["start"] < utt_end and tr["end"] > utt_start:
                intervals.append({
                    "startSec": round(tr["start"], 2),
                    "endSec": round(tr["end"], 2),
                    "maskType": mask_type,
                    "piiType": pii_item["type"],
                })
    return intervals


def curated_sequence_orders(rows: list[dict] | None) -> set[int]:
    """정책 P: 사람이 검수(pii_reviewed_at)하거나 마스킹(pii_masked_at)한 흔적이 있는
    sequence_order 집합. 이 행들의 pii_intervals 는 worker 가 덮어쓰지 않는다."""
    out: set[int] = set()
    for row in rows or []:
        if row.get("pii_reviewed_at") is not None or row.get("pii_masked_at") is not None:
            so = row.get("sequence_order")
            if so is not None:
                out.add(int(so))
    return out


def strip_pii_if_curated(
    row: dict,
    seq: int,
    curated_seqs: set[int],
    precheck_ok: bool,
) -> dict:
    """정책 P 적용: 해당 발화가 수동 검수/마스킹됐거나(curated) 사전조회가 실패(precheck_ok=False)하면
    upsert payload 에서 pii_intervals 키를 제거해 기존 값을 보존한다(나머지 STT/품질/라벨 컬럼은 갱신).
    precheck 실패 시 보수적으로 전 행 보존(사람 교정 손실 위험 > 자동 interval 미기록 비용)."""
    if not precheck_ok or seq in curated_seqs:
        row.pop("pii_intervals", None)
    return row


async def persist_results(session: dict, task_id: str, job_result: dict) -> int:
    """
    For each utterance: download WAV from voice_api → compute quality via ffprobe
    → upload to S3 → upsert to DB (conflict key: session_id, sequence_order).
    Also overwrites raw_audio_url with preprocessed audio (best-effort).
    Returns utterance count upserted.
    """
    session_id = session["id"]
    user_id = session["user_id"]
    utterances = job_result.get("utterances", [])
    loop = asyncio.get_running_loop()

    # ── STAGE 15: session_speakers ────────────────────────────────────────
    speaker_label_to_id: dict[str, str] = {}
    speakers_data = job_result.get("speakers", [])
    for spk in speakers_data:
        spk_row = {
            "session_id": session_id,
            "speaker_label": spk["speaker_label"],
            "speaker_role": spk.get("speaker_role"),
            "speaker_role_source": spk.get("speaker_role_source"),
            "speaker_gender": spk.get("speaker_gender"),
            "speaker_voice_age_range": spk.get("speaker_voice_age_range"),
            "speaker_speech_age_range": spk.get("speaker_speech_age_range"),
            "speaker_speech_age_model_version": spk.get("speaker_speech_age_model_version"),
            "speaker_relation": spk.get("speaker_relation"),
        }
        try:
            # NaN/Inf 가드: persist 경계에서 한 번 차단(현재는 string/enum payload 만이나 방어용).
            spk_row = sanitize_json_safe(spk_row)
            result = await _run(
                lambda r=spk_row: _supabase.table("session_speakers")
                .upsert(r, on_conflict="session_id,speaker_label")
                .execute()
            )
            if result.data:
                speaker_label_to_id[spk["speaker_label"]] = result.data[0]["id"]
        except Exception as e:
            log.warning("[%s] session_speakers upsert failed (%s): %s", session_id, spk["speaker_label"], e)

    pii_summary = job_result.get("pii_summary", [])

    # D4b 정책 P: 수동 검수/마스킹된 발화의 pii_intervals 는 worker 가 덮어쓰지 않는다.
    # upsert 전에 기존 행의 pii_reviewed_at/pii_masked_at 를 한 번에 조회해 보존 대상 seq 집합을 만든다.
    # 조회 실패 시 precheck_ok=False → 보수적으로 전 행의 pii_intervals 를 보존(덮어쓰기 회피).
    curated_seqs: set[int] = set()
    precheck_ok = True
    try:
        existing = await _run(
            lambda: _supabase.table("utterances")
            .select("sequence_order, pii_reviewed_at, pii_masked_at")
            .eq("session_id", session_id)
            .execute()
        )
        curated_seqs = curated_sequence_orders(existing.data)
    except Exception as e:
        precheck_ok = False
        log.warning(
            "[%s] pii curation precheck 실패 — 보수적으로 pii_intervals 덮어쓰기 보류: %s",
            session_id, e,
        )

    upserted = 0
    for i, utt in enumerate(utterances):
        seq: int = i + 1
        utt_id = f"utt_{session_id}_{str(seq).zfill(3)}"
        storage_path = f"utterances/{session_id}/{utt_id}.wav"
        audio_filename: str = utt.get("audio_filename", "")

        quality_score = None
        quality_grade = None
        snr_db = None
        speech_ratio = None
        file_size_bytes = None
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                tmp_path = tmp_f.name

            await download_utterance_wav(task_id, audio_filename, tmp_path)
            file_size_bytes = os.path.getsize(tmp_path)

            stats = await loop.run_in_executor(None, _get_audio_stats_sync, tmp_path)
            quality_score, quality_grade, snr_db, speech_ratio = _compute_quality(
                stats["rms_db"], stats["peak_db"], stats["silence_ratio"]
            )

            await _run(
                lambda p=tmp_path, sp=storage_path: _s3.upload_file(
                    p, S3_AUDIO_BUCKET, sp,
                    ExtraArgs={"ContentType": "audio/wav"},
                )
            )
        except Exception as e:
            log.warning("[%s] utt %s wav error: %s", session_id, utt_id, e)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        duration_sec = round(
            float(utt.get("end_sec") or 0) - float(utt.get("start_sec") or 0), 3
        )
        utt_start = float(utt.get("start_sec") or 0)
        utt_end = float(utt.get("end_sec") or 0)
        pii_intervals = build_pii_intervals(pii_summary, utt_start, utt_end)

        speaker_label: str | None = utt.get("speaker_id")
        row = {
            "id": utt_id,
            "session_id": session_id,
            "chunk_id": None,
            "user_id": user_id,
            "sequence_in_chunk": seq,
            "sequence_order": seq,
            "speaker_id": speaker_label,
            "session_speaker_id": speaker_label_to_id.get(speaker_label) if speaker_label else None,
            "is_user": False,
            "start_sec": utt.get("start_sec"),
            "end_sec": utt.get("end_sec"),
            "duration_sec": duration_sec,
            "storage_path": storage_path,
            "file_size_bytes": file_size_bytes,
            "upload_status": "uploaded",
            "transcript_text": utt.get("transcript_text", ""),
            "transcript_words": utt.get("words"),
            "segmented_by": "gpu_v10",
            "client_version": "gpu-worker-2.0",
            "quality_score": quality_score,
            "quality_grade": quality_grade,
            "snr_db": snr_db,
            "speech_ratio": speech_ratio,
            "pii_intervals": pii_intervals,
            # Stage 14: 자동 라벨
            "emotion": utt.get("emotion"),
            "emotion_confidence": utt.get("emotion_confidence"),
            "dialog_act": utt.get("dialog_act"),
            "dialog_act_confidence": utt.get("dialog_act_confidence"),
            "auto_label_model_version": utt.get("auto_label_model_version"),
            "label_source": utt.get("label_source"),
            # Stage 15 Tier A: 통계 라벨
            "speech_rate_wpm": utt.get("speech_rate_wpm"),
            "silence_before_sec": utt.get("silence_before_sec"),
            "filler_word_count": utt.get("filler_word_count"),
            "confidence_tier": utt.get("confidence_tier"),
            "audio_quality_class": utt.get("audio_quality_class"),
            # Stage 16 Tier B: 언어적 특성 라벨
            "honorific_level": utt.get("honorific_level"),
            "question_type": utt.get("question_type"),
            "language_mix_flag": utt.get("language_mix_flag"),
            "updated_at": _now_iso(),
        }
        # 정책 P: 수동 검수/마스킹된 행(또는 precheck 실패)이면 pii_intervals 를 payload 에서 제거해 보존.
        row = strip_pii_if_curated(row, seq, curated_seqs, precheck_ok)
        # NaN/Inf 가드: snr_db / transcript_words(WhisperX alignment) / *_confidence 등 모델 산출
        # 부동소수가 NaN/Inf 일 때 None 으로 치환(json.dumps allow_nan=False 통과 보장).
        row = sanitize_json_safe(row)
        await _run(
            lambda r=row: _supabase.table("utterances")
            .upsert(r, on_conflict="session_id,sequence_order")
            .execute()
        )
        upserted += 1

    # ── STAGE 16: session_segments ────────────────────────────────────────
    topic_segments_data = job_result.get("topic_segments", [])
    segment_index_to_id: dict[int, str] = {}
    for seg in topic_segments_data:
        seg_row = {
            "session_id": session_id,
            "segment_index": seg["segment_index"],
            "topic": seg.get("topic"),
            "start_ms": seg.get("start_ms"),
            "end_ms": seg.get("end_ms"),
            "utterance_count": len(seg.get("utterance_indices", [])),
        }
        try:
            # NaN/Inf 가드: 현재 payload 는 int/str 만이나 방어용으로 persist 경계에서 차단.
            seg_row = sanitize_json_safe(seg_row)
            result = await _run(
                lambda r=seg_row: _supabase.table("session_segments")
                .upsert(r, on_conflict="session_id,segment_index")
                .execute()
            )
            if result.data:
                segment_index_to_id[seg["segment_index"]] = result.data[0]["id"]
        except Exception as e:
            log.warning("[%s] session_segments upsert failed (%d): %s", session_id, seg["segment_index"], e)

    for seg in topic_segments_data:
        seg_id = segment_index_to_id.get(seg["segment_index"])
        if not seg_id:
            continue
        for utt_idx in seg.get("utterance_indices", []):
            utt_seq = utt_idx + 1
            utt_seg_id = f"utt_{session_id}_{str(utt_seq).zfill(3)}"
            try:
                await _run(
                    lambda uid=utt_seg_id, sid=seg_id: _supabase.table("utterances")
                    .update({"segment_id": sid})
                    .eq("id", uid)
                    .execute()
                )
            except Exception as e:
                log.warning("[%s] utterances.segment_id update failed (%s): %s", session_id, utt_seg_id, e)

    # Overwrite raw_audio_url with preprocessed audio (best-effort, skip on error)
    preproc_tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            preproc_tmp = f.name
        await download_utterance_wav(task_id, "_preprocessed_audio.wav", preproc_tmp)
        raw_key = session["raw_audio_url"]
        await _run(
            lambda: _s3.upload_file(
                preproc_tmp, S3_AUDIO_BUCKET, raw_key,
                ExtraArgs={"ContentType": "audio/wav"},
            )
        )
    except Exception as e:
        log.debug("[%s] preprocessed audio overwrite skipped: %s", session["id"], e)
    finally:
        if preproc_tmp and os.path.exists(preproc_tmp):
            os.unlink(preproc_tmp)

    return upserted


# ── Error handling ────────────────────────────────────────────────────

async def increment_retry(session_id: str, error_msg: str) -> None:
    """
    Increment gpu_retry_count via Supabase RPC (atomic).
    Fallback: SELECT current count → UPDATE count+1 if RPC unavailable.
    Mirrors gpu-worker.ts:569 supabaseAdmin.rpc('increment_gpu_retry', ...).
    """
    truncated = error_msg[:2000]
    try:
        await _run(
            lambda: _supabase.rpc("increment_gpu_retry", {
                "p_session_id": session_id,
                "p_error_msg": truncated,
            }).execute()
        )
    except Exception as rpc_err:
        log.warning("[%s] increment_gpu_retry RPC failed (%s), fallback", session_id, rpc_err)
        try:
            result = await _run(
                lambda: _supabase.table("sessions")
                .select("gpu_retry_count")
                .eq("id", session_id)
                .execute()
            )
            current = int((result.data or [{}])[0].get("gpu_retry_count") or 0)
            await _run(
                lambda: _supabase.table("sessions").update({
                    "gpu_upload_status": "failed",
                    "gpu_retry_count": current + 1,
                    "gpu_last_error": truncated,
                    "updated_at": _now_iso(),
                }).eq("id", session_id).execute()
            )
        except Exception as fallback_err:
            log.error("[%s] fallback retry increment failed: %s", session_id, fallback_err)


# ── Stuck session sweep ───────────────────────────────────────────────

async def sweep_stuck_sessions() -> None:
    """Force-fail sessions stuck in 'running' for > STUCK_THRESHOLD_SEC (10 min)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=STUCK_THRESHOLD_SEC)).isoformat()
    try:
        result = await _run(
            lambda: _supabase.table("sessions").select("id")
            .eq("gpu_upload_status", "running")
            .lt("gpu_started_at", cutoff)
            .execute()
        )
        for row in result.data or []:
            sid = row["id"]
            log.warning("sweep: forcing stuck session %s → failed", sid)
            await _run(
                lambda s=sid: _supabase.table("sessions").update({
                    "gpu_upload_status": "failed",
                    "gpu_last_error": "worker timeout (stuck > 10min)",
                    "updated_at": _now_iso(),
                }).eq("id", s).execute()
            )
    except Exception as e:
        log.error("sweep_stuck_sessions error: %s", e)

    # Feature 1: both_agreed + raw_audio_url IS NULL 영구 stall 경보
    try:
        null_audio_res = await _run(
            lambda: _supabase.table("sessions").select("id", count="exact")
            .eq("consent_status", "both_agreed")
            .is_("raw_audio_url", "null")
            .neq("gpu_upload_status", "done")
            .neq("gpu_upload_status", "skipped")
            .execute()
        )
        null_count = null_audio_res.count or 0
        if null_count > 5:
            log.warning(
                "sweep: %d sessions both_agreed + raw_audio_url IS NULL (영구 stall 의심) — 수동 조치 필요",
                null_count,
            )
    except Exception as e:
        log.error("sweep null_audio_url check error: %s", e)


async def sweep_segment_backfill() -> None:
    """Feature 2: quality_status=done 인데 segment_id=NULL 발화를 시간 범위로 역할당."""
    try:
        orphan_res = await _run(
            lambda: _supabase.table("utterances")
            .select("id, session_id, start_sec")
            .is_("segment_id", "null")
            .limit(200)
            .execute()
        )
        orphans = orphan_res.data or []
        if not orphans:
            return

        sessions_to_fix: dict[str, list[dict]] = {}
        for utt in orphans:
            sessions_to_fix.setdefault(utt["session_id"], []).append(utt)

        patched = 0
        for session_id, utts in sessions_to_fix.items():
            seg_res = await _run(
                lambda s=session_id: _supabase.table("session_segments")
                .select("id, start_ms, end_ms")
                .eq("session_id", s)
                .order("start_ms")
                .execute()
            )
            segments = seg_res.data or []
            if not segments:
                continue

            for utt in utts:
                utt_ms = (utt.get("start_sec") or 0) * 1000
                matched_seg_id = None
                for seg in segments:
                    if seg["start_ms"] <= utt_ms <= seg["end_ms"]:
                        matched_seg_id = seg["id"]
                        break
                if not matched_seg_id:
                    nearest = min(segments, key=lambda s: abs(s["start_ms"] - utt_ms))
                    matched_seg_id = nearest["id"]
                try:
                    await _run(
                        lambda uid=utt["id"], sid=matched_seg_id: _supabase.table("utterances")
                        .update({"segment_id": sid})
                        .eq("id", uid)
                        .execute()
                    )
                    patched += 1
                except Exception as e:
                    log.warning("[%s] backfill segment_id failed for %s: %s", session_id, utt["id"], e)

        if patched:
            log.info("sweep_segment_backfill: %d orphan utterances patched in %d sessions",
                     patched, len(sessions_to_fix))
    except Exception as e:
        log.error("sweep_segment_backfill error: %s", e)


# ── Session processing ────────────────────────────────────────────────

async def process_one_session() -> str:
    """
    Pick and fully process one session.
    Returns 'done' (success or error handled), 'empty' (nothing to do), or '503'.
    """
    session = await pick_next_session()
    if not session:
        return "empty"

    session_id = session["id"]
    log.info("[%s] start processing", session_id)
    audio_path = None

    try:
        audio_path = await download_raw_audio(session["raw_audio_url"])
        log.info("[%s] audio downloaded to %s", session_id, audio_path)

        task_id = await submit_to_voice_api(audio_path)
        log.info("[%s] submitted → task_id %s", session_id, task_id)

        job_result = await poll_job(task_id)
        log.info("[%s] job completed", session_id)

        utterance_count = await persist_results(session, task_id, job_result)
        log.info("[%s] persisted %d utterances", session_id, utterance_count)

        # Feature 3: both_agreed + 화자 1명 + 발화 6개 이상 → 재처리 큐
        if (
            session.get("consent_status") == "both_agreed"
            and utterance_count >= 6
            and not (session.get("gpu_last_error") or "").startswith("SPEAKER_REQUEUE:")
        ):
            sp_res = await _run(
                lambda: _supabase.table("session_speakers")
                .select("id", count="exact")
                .eq("session_id", session_id)
                .execute()
            )
            if (sp_res.count or 0) == 1:
                log.warning(
                    "[%s] both_agreed + 화자 1명 + %d 발화 — pyannote 실패 의심, 재처리 큐 진입",
                    session_id, utterance_count,
                )
                await _run(
                    lambda: _supabase.table("sessions").update({
                        "gpu_upload_status": "pending",
                        "gpu_started_at": None,
                        "gpu_retry_count": 0,
                        "gpu_last_error": f"SPEAKER_REQUEUE: both_agreed+1speaker+{utterance_count}utts",
                        "updated_at": _now_iso(),
                    }).eq("id", session_id).execute()
                )
                return "done"

        now = _now_iso()
        auto_label_status = "done" if utterance_count > 0 else "skipped"
        quality_status = "done" if utterance_count > 0 else "skipped"

        await _run(
            lambda: _supabase.table("sessions").update({
                "gpu_upload_status": "done",
                "stt_status": "done",
                "stt_at": now,
                "diarize_status": "done",
                "diarize_at": now,
                "gpu_pii_status": "done",
                "gpu_pii_at": now,
                "auto_label_status": auto_label_status,
                "quality_status": quality_status,
                "quality_at": now,
                "utterance_count": utterance_count,
                "gpu_last_error": None,
                "updated_at": now,
            }).eq("id", session_id).execute()
        )
        log.info("[%s] done (%d utterances)", session_id, utterance_count)
        return "done"

    except Voice503Error as e:
        log.warning("[%s] 503 from voice_api: %s", session_id, e)
        # Reset to pending — NO retry_count increment (matches POLL_BACKOFF_503_MS policy)
        await _run(
            lambda: _supabase.table("sessions").update({
                "gpu_upload_status": "pending",
                "gpu_started_at": None,
                "gpu_last_error": str(e)[:2000],
                "updated_at": _now_iso(),
            }).eq("id", session_id).execute()
        )
        return "503"

    except Exception as e:
        log.error("[%s] processing failed: %s", session_id, e, exc_info=True)
        await increment_retry(session_id, str(e)[:2000])
        return "done"

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


# ── Worker loops ──────────────────────────────────────────────────────

async def poll_loop(worker_index: int) -> None:
    log.info("worker %d started", worker_index)
    while not is_shutting_down:
        result = await process_one_session()
        if result == "503":
            log.info("worker %d: 503 backoff %ds", worker_index, POLL_BACKOFF_503_SEC)
            await asyncio.sleep(POLL_BACKOFF_503_SEC)
        elif result == "empty":
            await asyncio.sleep(POLL_INTERVAL_SEC)
        # result == "done" → immediate next pick


async def sweep_loop() -> None:
    while not is_shutting_down:
        await asyncio.sleep(STUCK_SWEEP_INTERVAL_SEC)
        await sweep_stuck_sessions()
        await sweep_segment_backfill()


# ── Entry point ───────────────────────────────────────────────────────

async def main() -> None:
    global _supabase, _s3, _http, is_shutting_down

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    log.info(
        "GPU worker starting — CONCURRENCY=%d, VOICE_API_URL=%s",
        WORKER_CONCURRENCY, VOICE_API_URL,
    )

    _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    _s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    _http = aiohttp.ClientSession()

    loop = asyncio.get_running_loop()

    def _on_signal(sig):
        global is_shutting_down
        log.info("received %s — shutting down gracefully", sig.name)
        is_shutting_down = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, functools.partial(_on_signal, sig))

    await asyncio.gather(
        sweep_loop(),
        *[poll_loop(i) for i in range(WORKER_CONCURRENCY)],
    )

    await _http.close()
    log.info("GPU worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
