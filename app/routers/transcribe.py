import json
import logging
import time
import uuid
from pathlib import Path as FilePath, PurePosixPath
from typing import Union

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import FileResponse, Response

from app import config
from app.core.job_store import job_store
from app.models.schemas import (
    ErrorResponse,
    JobPendingResponse,
    TaskStatus,
    TranscribeAcceptedResponse,
    TranscribeResultResponse,
)
from app.services.whisperx_service import whisperx_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["transcribe"])


def _process_audio(
    task_id: str,
    file_path: str,
    diarize: bool,
    enable_name_masking: bool,
    mask_pii: bool = True,
    split_by_speaker: bool = False,
    split_by_utterance: bool = False,
    denoise: bool | None = None,
    mask_audio_pii: bool = False,
    mask_audio_names: bool = False,
    reference_embedding: str | None = None,
):
    """Background task: run WhisperX pipeline and update job store."""
    ref_emb: list[float] | None = None
    if reference_embedding:
        try:
            ref_emb = json.loads(reference_embedding)
        except (json.JSONDecodeError, ValueError):
            logger.warning("[%s] reference_embedding JSON 파싱 실패 — 무시", task_id)
    try:
        job_store.update_status(task_id, TaskStatus.processing)
        result = whisperx_service.transcribe(
            file_path=file_path,
            task_id=task_id,
            enable_diarize=diarize,
            enable_name_masking=enable_name_masking,
            mask_pii=mask_pii,
            split_by_speaker=split_by_speaker,
            split_by_utterance=split_by_utterance,
            denoise_enabled=denoise,
            mask_audio_pii=mask_audio_pii,
            mask_audio_names=mask_audio_names,
            reference_embedding=ref_emb,
        )
        audio_files = result.pop("_audio_files", None)
        if audio_files:
            logger.info("[%s] 오디오 %d개 파일 저장", task_id, len(audio_files))
            job_store.set_audio(task_id, audio_files)
        logger.info("[%s] set_result 호출 (keys: %s)", task_id, list(result.keys()))
        job_store.set_result(task_id, result)
        # Verify — race가 발생하면 set_result가 무시했을 수 있음 (terminal guard / evicted)
        stored = job_store.get(task_id)
        if stored is None or stored.result is None:
            logger.error(
                "[%s] CRITICAL: set_result 후 결과 부재 — race 의심. "
                "stored=%s result_keys=%s",
                task_id,
                stored.status.value if stored else "NONE",
                list(stored.result.keys()) if stored and stored.result else "NONE",
            )
        else:
            logger.info(
                "[%s] 저장 검증: status=%s result_keys=%s",
                task_id, stored.status.value, list(stored.result.keys()),
            )
    except Exception as e:
        logger.error("[%s] Processing failed: %s", task_id, e)
        job_store.set_error(task_id, str(e))


@router.post(
    "/transcribe",
    response_model=TranscribeAcceptedResponse,
    status_code=200,
    summary="음성 파일 업로드 및 STT 요청",
    description=(
        "음성 파일을 업로드하면 비동기 STT 작업을 생성합니다.\n\n"
        "### 지원 포맷\n\n"
        "`wav` · `mp3` · `m4a` · `ogg` · `flac` · `webm` · `mp4`\n\n"
        "### 최대 업로드 크기\n\n"
        "**500MB** (초과 시 `413` 반환)\n\n"
        "### 처리 흐름\n\n"
        "1. 파일을 RAM 디스크(`/dev/shm`)에 임시 저장\n"
        "2. 백그라운드에서 WhisperX 파이프라인 실행\n"
        "3. 처리 완료 후 원본 음성 파일 자동 삭제\n\n"
        "### 결과 조회\n\n"
        "반환된 `task_id`로 `GET /api/v1/jobs/{task_id}`를 폴링하세요.\n"
        "권장 폴링 간격: **1~2초**"
    ),
    responses={
        200: {
            "description": "업로드 성공 — 작업이 등록되어 백그라운드 처리 시작",
            "model": TranscribeAcceptedResponse,
        },
        400: {
            "description": "지원하지 않는 파일 포맷",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Unsupported format: txt"},
                },
            },
        },
        413: {
            "description": "파일 크기 초과 (500MB 제한)",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "File too large"},
                },
            },
        },
    },
)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ...,
        description="음성 파일. 지원 확장자: wav, mp3, m4a, ogg, flac, webm, mp4",
    ),
    language: str = Query(
        "ko",
        description="인식 대상 언어 코드 (ISO 639-1). 기본값 `ko` (한국어)",
        examples=["ko", "en", "ja"],
    ),
    diarize: bool = Query(
        False,
        description="화자분리 활성화 여부. `true`로 설정하면 세그먼트에 `speaker` 필드가 추가됩니다. "
        "서버에 `HF_TOKEN` 환경변수가 설정되어 있어야 합니다.",
    ),
    mask_pii: bool = Query(
        True,
        description="PII 자동 마스킹 활성화 여부. `true`(기본)이면 주민등록번호, "
        "전화번호, 이메일 등 9종 PII를 자동 마스킹합니다.",
    ),
    enable_name_masking: bool = Query(
        False,
        description="한국어 이름 마스킹 활성화 여부. `true`로 설정하면 "
        "한국 성씨(40종) + 이름 패턴을 탐지하여 `김OO` 형태로 마스킹합니다. "
        "문맥 기반 휴리스틱을 사용하므로 일부 오탐/미탐이 있을 수 있습니다.",
    ),
    split_by_speaker: bool = Query(
        False,
        description="화자별 오디오 분리 활성화. `diarize=true` 필수. "
        "각 화자의 음성을 별도 WAV 파일로 저장합니다.",
    ),
    split_by_utterance: bool = Query(
        False,
        description="발화별 오디오 분리 활성화. `diarize=true` 필수. "
        "각 발화 구간을 별도 WAV 파일로 저장합니다.",
    ),
    denoise: bool = Query(
        True,
        description="배경 잡음 제거(DeepFilterNet) 활성화 여부. `false`로 설정하면 "
        "깨끗한 오디오의 전처리 속도를 개선할 수 있습니다.",
    ),
    mask_audio_pii: bool = Query(
        False,
        description="음성 PII 마스킹 활성화 여부. `true`로 설정하면 "
        "감지된 PII 구간을 1kHz 비프음으로 치환합니다.",
    ),
    mask_audio_names: bool = Query(
        False,
        description="음성 내 이름 마스킹 활성화 여부. `mask_audio_pii=true`일 때만 유효하며, "
        "한국 성씨+이름 패턴을 탐지하여 비프음 처리합니다.",
    ),
    reference_embedding: str | None = Query(
        None,
        description="발화자 voice profile 임베딩 (JSON float 배열). "
        "uncounted-api가 voice_profiles에서 조회하여 전달. "
        "diarize=true일 때만 유효하며, 전달 시 가장 유사한 화자를 self로 식별합니다.",
    ),
):
    # 확장자 검증
    ext = FilePath(file.filename or "").suffix.lstrip(".").lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        # 파일명은 전화번호·인명 PII를 포함할 수 있어 평문 로깅 금지. 확장자 + 길이만 기록.
        logger.warning(
            "[reject-400] ext=%r filename_len=%d",
            ext,
            len(file.filename or ""),
        )
        raise HTTPException(400, f"Unsupported format: {ext}")

    # 큐 백프레셔: pending + processing 합산이 한계 이상이면 503 반환
    active = job_store.active_count()
    if active >= config.MAX_ACTIVE_JOBS:
        retry_after = config.QUEUE_FULL_RETRY_AFTER_SEC
        logger.warning(
            "[queue-full] active=%d limit=%d — 503 reject", active, config.MAX_ACTIVE_JOBS
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "queue_full",
                "active_jobs": active,
                "max_active_jobs": config.MAX_ACTIVE_JOBS,
                "retry_after_sec": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    # 임시 저장 경로 준비
    # task_id = uuid 8자(32-bit) + microsecond 6자 = 14자.
    # 12자 hex 단독 시 동시 100+ 요청에서 collision 이론적 가능 — timestamp로 사실상 불가능화.
    task_id = f"{uuid.uuid4().hex[:8]}{int(time.time() * 1_000_000) % 1_000_000:06d}"
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    file_path = config.TEMP_DIR / f"{task_id}.{ext}"

    # 청크 스트리밍으로 파일 작성 및 크기 추적
    chunk_size = 65536  # 64KB chunks
    total_size = 0
    size_exceeded = False
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > config.MAX_UPLOAD_SIZE:
                    size_exceeded = True
                    break
                await f.write(chunk)
    except Exception as e:
        logger.error("[%s] Upload failed: %s", task_id, e)
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, "Upload failed")

    if size_exceeded:
        file_path.unlink(missing_ok=True)
        raise HTTPException(413, "File too large")

    # 작업 등록 + 백그라운드 실행
    job_store.create(task_id)
    background_tasks.add_task(
        _process_audio,
        task_id,
        str(file_path),
        diarize,
        enable_name_masking,
        mask_pii,
        split_by_speaker,
        split_by_utterance,
        denoise,
        mask_audio_pii,
        mask_audio_names,
        reference_embedding,
    )

    return {"task_id": task_id, "status": "pending"}


@router.get(
    "/jobs/{task_id}",
    response_model=None,
    summary="작업 상태 및 결과 조회",
    description=(
        "작업 ID로 STT 처리 상태를 조회합니다.\n\n"
        "### 상태별 응답\n\n"
        "| 상태 | HTTP | 응답 내용 |\n"
        "|------|------|-----------|\n"
        "| `pending` | 200 | `task_id` + `status` |\n"
        "| `processing` | 200 | `task_id` + `status` |\n"
        "| `completed` | 200 | 전체 결과 (세그먼트, PII 요약, 발화분리, 화자별 오디오) |\n"
        "| `failed` | 500 | 에러 메시지 |\n\n"
        "### 폴링 권장 사항\n\n"
        "- 간격: **1~2초**\n"
        "- 최대 대기: 음성 길이에 비례 (1분 음성 ≈ 5~15초 처리)\n"
        "- `completed` 또는 `failed` 수신 시 폴링 중단"
    ),
    responses={
        200: {
            "description": "작업 진행 중이면 상태만, 완료 시 전체 결과 반환",
            "model": TranscribeResultResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "진행 중 (pending/processing)": {
                            "summary": "작업 대기/처리 중",
                            "value": {"task_id": "a1b2c3d4e5f6", "status": "processing"},
                        },
                        "완료 — 기본 (화자분리 없음)": {
                            "summary": "STT 완료 — 기본 응답",
                            "value": {
                                "task_id": "a1b2c3d4e5f6",
                                "status": "completed",
                                "language": "ko",
                                "duration_seconds": 4.72,
                                "segments": [
                                    {"start": 0.0, "end": 3.52, "text": "안녕하세요", "speaker": None, "words": None},
                                    {"start": 3.52, "end": 6.10, "text": "김OO입니다", "speaker": None, "words": None},
                                ],
                                "full_text": "안녕하세요 김OO입니다",
                                "pii_summary": [{"type": "이름", "count": 1}],
                                "diarization_enabled": False,
                                "utterances": None,
                                "speaker_audio": None,
                            },
                        },
                        "완료 — 화자분리 + words": {
                            "summary": "STT 완료 — 화자분리 + word-level 타임스탬프",
                            "value": {
                                "task_id": "b2c3d4e5f6a1",
                                "status": "completed",
                                "language": "ko",
                                "duration_seconds": 20.17,
                                "segments": [
                                    {
                                        "start": 0.03,
                                        "end": 1.19,
                                        "text": "왜 그러지?",
                                        "speaker": "SPEAKER_02",
                                        "words": [
                                            {"word": "왜", "start": 0.03, "end": 0.59, "speaker": "SPEAKER_02"},
                                            {"word": "그러지?", "start": 0.73, "end": 1.19, "speaker": "SPEAKER_00"},
                                        ],
                                    },
                                ],
                                "full_text": "왜 그러지? 어 끊어졌네 갑자기 ...",
                                "pii_summary": [{"type": "이름", "count": 1}],
                                "diarization_enabled": True,
                                "utterances": None,
                                "speaker_audio": None,
                            },
                        },
                        "완료 — 전체 옵션 (발화분리 + 화자별 오디오)": {
                            "summary": "STT 완료 — diarize + split_by_utterance + split_by_speaker 전체 적용",
                            "value": {
                                "task_id": "c3d4e5f6a1b2",
                                "status": "completed",
                                "language": "ko",
                                "duration_seconds": 20.17,
                                "segments": [
                                    {
                                        "start": 0.03,
                                        "end": 1.19,
                                        "text": "왜 그러지?",
                                        "speaker": "SPEAKER_02",
                                        "words": [
                                            {"word": "왜", "start": 0.03, "end": 0.59, "speaker": "SPEAKER_02"},
                                            {"word": "그러지?", "start": 0.73, "end": 1.19, "speaker": "SPEAKER_00"},
                                        ],
                                    },
                                ],
                                "full_text": "왜 그러지? 어 끊어졌네 갑자기 ...",
                                "pii_summary": [{"type": "이름", "count": 1}],
                                "diarization_enabled": True,
                                "utterances": [
                                    {
                                        "index": 0,
                                        "start_sec": 0.03,
                                        "end_sec": 2.79,
                                        "duration_sec": 2.76,
                                        "speaker_id": "SPEAKER_02",
                                        "transcript_text": "왜 그러지? 어 끊어졌네 갑자기",
                                        "audio_filename": "utterance_000.wav",
                                        "words": [
                                            {"word": "왜", "start": 0.03, "end": 0.59, "speaker": "SPEAKER_02"},
                                            {"word": "그러지?", "start": 0.73, "end": 1.19, "speaker": "SPEAKER_00"},
                                        ],
                                    },
                                    {
                                        "index": 1,
                                        "start_sec": 3.99,
                                        "end_sec": 10.50,
                                        "duration_sec": 6.51,
                                        "speaker_id": "SPEAKER_01",
                                        "transcript_text": "이게 엘리베이터만 타면 끊기더라고",
                                        "audio_filename": "utterance_001.wav",
                                        "words": [
                                            {"word": "이게", "start": 3.99, "end": 4.29, "speaker": "SPEAKER_01"},
                                        ],
                                    },
                                ],
                                "speaker_audio": [
                                    {
                                        "speaker_id": "SPEAKER_01",
                                        "total_duration_sec": 246.91,
                                        "audio_filename": "speaker_speaker_01.wav",
                                    },
                                    {
                                        "speaker_id": "SPEAKER_02",
                                        "total_duration_sec": 246.91,
                                        "audio_filename": "speaker_speaker_02.wav",
                                    },
                                ],
                            },
                        },
                    },
                },
            },
        },
        404: {
            "description": "존재하지 않는 task_id",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Task not found"},
                },
            },
        },
        500: {
            "description": "STT 처리 실패",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Processing failed: CUDA out of memory"},
                },
            },
        },
    },
)
async def get_job_status(
    task_id: str = Path(
        ...,
        description="업로드 시 반환받은 작업 ID",
        examples=["a1b2c3d4140000"],
        min_length=12,
        max_length=14,
    ),
):
    task = job_store.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")

    if task.status == TaskStatus.failed:
        raise HTTPException(500, f"Processing failed: {task.error}")

    if task.status == TaskStatus.completed:
        return task.result

    # 진행 중 응답에 관측용 필드 동봉 (폴링 타임아웃 원인 분석용)
    snapshot = job_store.queue_snapshot()
    position = job_store.position_of(task_id)
    now = time.time()

    if task.queued_at is not None:
        if task.gpu_acquired_at is not None:
            elapsed_queue = task.gpu_acquired_at - task.queued_at
        else:
            elapsed_queue = now - task.queued_at
    else:
        elapsed_queue = None

    if task.gpu_acquired_at is not None:
        end_ts = task.gpu_released_at if task.gpu_released_at is not None else now
        elapsed_processing = end_ts - task.gpu_acquired_at
    else:
        elapsed_processing = None

    return {
        "task_id": task_id,
        "status": task.status.value,
        "position_in_queue": position,
        "queue_size": snapshot["queue_depth"] + (1 if snapshot["gpu_busy"] else 0),
        "gpu_busy": snapshot["gpu_busy"],
        "queued_at": task.queued_at,
        "gpu_acquired_at": task.gpu_acquired_at,
        "elapsed_queue_seconds": (
            round(elapsed_queue, 2) if elapsed_queue is not None else None
        ),
        "elapsed_processing_seconds": (
            round(elapsed_processing, 2) if elapsed_processing is not None else None
        ),
    }


@router.get(
    "/jobs/{task_id}/audio/{filename}",
    summary="분리된 오디오 바이너리 다운로드",
    description=(
        "화자별 또는 발화별로 분리된 WAV 오디오를 바이너리로 응답합니다.\n\n"
        "인메모리에서 직접 서빙되며, 서버 재시작 시 소실됩니다.\n\n"
        "### 파일명 규칙\n\n"
        "| 유형 | 패턴 | 예시 |\n"
        "|------|------|------|\n"
        "| 발화별 | `utterance_{NNN}.wav` | `utterance_000.wav` |\n"
        "| 화자별 | `speaker_{SPEAKER_ID}.wav` | `speaker_speaker_01.wav` |"
    ),
    responses={
        200: {
            "description": "WAV 바이너리 (`Content-Type: audio/wav`)",
            "content": {"audio/wav": {}},
        },
        404: {
            "description": "task 또는 파일 없음",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "task 없음": {
                            "summary": "task_id가 존재하지 않거나 미완료",
                            "value": {"detail": "Task not found or not completed"},
                        },
                        "파일 없음": {
                            "summary": "요청한 오디오 파일이 존재하지 않음",
                            "value": {"detail": "Audio not found: utterance_999.wav"},
                        },
                    },
                },
            },
        },
    },
)
async def download_split_audio(
    task_id: str = Path(
        ...,
        description="작업 ID",
        examples=["a1b2c3d4140000"],
        min_length=12,
        max_length=14,
    ),
    filename: str = Path(
        ...,
        description="다운로드할 오디오 파일명 (utterance_NNN.wav 또는 speaker_SPEAKER_ID.wav)",
        examples=["utterance_000.wav", "speaker_speaker_01.wav"],
    ),
):
    task = job_store.get(task_id)
    if task is None or task.status != TaskStatus.completed:
        raise HTTPException(404, "Task not found or not completed")

    audio_path = job_store.get_audio(task_id, filename)
    if audio_path is None:
        raise HTTPException(404, f"Audio not found: {filename}")

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=filename,
    )
