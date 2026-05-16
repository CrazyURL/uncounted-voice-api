"""
GPU Worker — Python port of gpu-worker.ts
Polls Supabase for pending audio sessions and processes them via voice_api.
Run as a standalone process on the GPU server alongside uncounted-voice_api.
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
from supabase import create_client, Client

# TODO(post-seed): worker_heartbeats 테이블 + 어드민 5분 무응답 알람

# ── Constants (matched 1:1 with gpu-worker.ts) ────────────────────────
POLL_INTERVAL_SEC = 30
POLL_BACKOFF_503_SEC = 60           # 503 → 60s backoff, NO retry_count increment
STUCK_SWEEP_INTERVAL_SEC = 300      # 5 min
STUCK_THRESHOLD_SEC = 600           # 10 min
RETRY_DELAY_SEC = 1800              # 30 min between retries
MAX_RETRY_COUNT = 3
VOICE_API_POLL_INTERVAL_SEC = 1
VOICE_API_MAX_WAIT_SEC = 300        # 5 min

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
        .order("created_at")
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
    """Stream S3 object to a temp file on disk (never in RAM). Returns path — caller deletes."""
    ext = raw_audio_url.rsplit(".", 1)[-1].lower() if "." in raw_audio_url else "wav"
    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        await _run(lambda: _s3.download_fileobj(S3_AUDIO_BUCKET, raw_audio_url, tmp))
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


# ── Voice API ─────────────────────────────────────────────────────────

async def submit_to_voice_api(audio_path: str) -> str:
    """POST audio file to voice_api; return task_id. 503 → Voice503Error."""
    params = {
        "language": "ko",
        "diarize": "true",
        "split_by_utterance": "true",
        "mask_pii": "true",
        "denoise": "true",
    }
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
    """Blocking: run ffprobe to extract RMS/Peak dB and silence ratio."""
    # RMS and Peak levels via astats filter
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-af", "astats=metadata=1:reset=0",
         "-f", "null", "-i", wav_path],
        capture_output=True, text=True,
    )
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
        ["ffprobe", "-v", "error", "-af", "silencedetect=noise=-40dB:d=0.3",
         "-f", "null", "-i", wav_path],
        capture_output=True, text=True,
    )
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
    """Port of qualityMetricsService.ts computeQualityScore. Returns (score, grade)."""
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
    return score, grade


# ── Persist results ───────────────────────────────────────────────────

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

    upserted = 0
    for i, utt in enumerate(utterances):
        seq: int = i + 1
        utt_id = f"utt_{session_id}_{str(seq).zfill(3)}"
        storage_path = f"utterances/{session_id}/{utt_id}.wav"
        audio_filename: str = utt.get("audio_filename", "")

        quality_score = None
        quality_grade = None
        file_size_bytes = None
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                tmp_path = tmp_f.name

            await download_utterance_wav(task_id, audio_filename, tmp_path)
            file_size_bytes = os.path.getsize(tmp_path)

            stats = await loop.run_in_executor(None, _get_audio_stats_sync, tmp_path)
            quality_score, quality_grade = _compute_quality(
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
        row = {
            "id": utt_id,
            "session_id": session_id,
            "chunk_id": None,
            "user_id": user_id,
            "sequence_in_chunk": seq,
            "sequence_order": seq,
            "speaker_id": utt.get("speaker_id"),
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
            "updated_at": _now_iso(),
        }
        await _run(
            lambda r=row: _supabase.table("utterances")
            .upsert(r, on_conflict="session_id,sequence_order")
            .execute()
        )
        upserted += 1

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
