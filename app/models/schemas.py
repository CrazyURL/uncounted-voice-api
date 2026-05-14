from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """STT 작업의 처리 상태."""

    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TaskInfo(BaseModel):
    """내부 작업 상태 관리용 모델 (API 응답에 직접 노출되지 않음).

    관측용 timing 필드(queued_at / gpu_acquired_at / gpu_released_at)는 유닉스
    초 단위 timestamp이며, 폴링 타임아웃 원인 분석용이다.
    """
    model_config = {"arbitrary_types_allowed": True}

    task_id: str
    status: TaskStatus = TaskStatus.pending
    result: Any = None
    error: Optional[str] = None
    queued_at: Optional[float] = None
    gpu_acquired_at: Optional[float] = None
    gpu_released_at: Optional[float] = None


class TranscribeRequest(BaseModel):
    """음성 파일 업로드 시 쿼리 파라미터 (참조용 — 실제는 Query params로 전달)."""

    language: str = "ko"
    diarize: bool = False
    mask_pii: bool = True
    enable_name_masking: bool = False


class WordResponse(BaseModel):
    """단어 단위 타임스탬프 및 화자 정보.

    WhisperX Forced Alignment으로 산출된 word-level 데이터입니다.
    """

    word: str = Field(
        ...,
        description="인식된 단어 텍스트",
        examples=["안녕하세요"],
    )
    start: float = Field(
        ...,
        description="단어 시작 시간 (초)",
        examples=[0.03],
        ge=0,
    )
    end: float = Field(
        ...,
        description="단어 종료 시간 (초)",
        examples=[1.19],
        ge=0,
    )
    speaker: Optional[str] = Field(
        None,
        description="화자 ID. `diarize=true` 시에만 포함",
        examples=["SPEAKER_00"],
    )


class SegmentResponse(BaseModel):
    """타임스탬프 기반 개별 세그먼트.

    WhisperX의 Forced Alignment으로 단어 단위 정밀 시간이 산출되며,
    화자분리 활성화 시 `speaker` 필드가 추가됩니다.
    """

    start: float = Field(
        ...,
        description="세그먼트 시작 시간 (초, 소수점 2자리)",
        examples=[0.0],
        ge=0,
    )
    end: float = Field(
        ...,
        description="세그먼트 종료 시간 (초, 소수점 2자리)",
        examples=[3.52],
        ge=0,
    )
    text: str = Field(
        ...,
        description="PII 마스킹이 적용된 텍스트",
        examples=["안녕하세요 김OO입니다"],
    )
    speaker: Optional[str] = Field(
        None,
        description="화자 ID. `diarize=true` 요청 시에만 포함됩니다. pyannote 기반으로 자동 할당.",
        examples=["SPEAKER_00"],
    )
    words: Optional[list[WordResponse]] = Field(
        None,
        description="단어 단위 타임스탬프 목록. `diarize=true` 시 각 word에 speaker 포함",
    )


class TimeRange(BaseModel):
    """오디오 시간 범위 (초)."""

    start: float = Field(..., description="시작 시간 (초)", examples=[1.23])
    end: float = Field(..., description="종료 시간 (초)", examples=[4.56])


class PIIDetectedItem(BaseModel):
    """감지된 PII 유형별 요약.

    마스킹 대상 9종: 주민등록번호, 운전면허번호, 여권번호, 카드번호,
    이메일, 전화번호, 계좌번호, IP주소, 이름(선택).
    """

    type: str = Field(
        ...,
        description="PII 유형 (주민등록번호 | 운전면허번호 | 여권번호 | 카드번호 | "
        "이메일 | 전화번호 | 계좌번호 | IP주소 | 이름)",
        examples=["전화번호"],
    )
    count: int = Field(
        ...,
        description="해당 유형의 감지 건수",
        examples=[2],
        ge=1,
    )
    time_ranges: Optional[list[TimeRange]] = Field(
        None,
        description="해당 PII가 감지된 오디오 시간 범위 목록. `mask_audio_pii=true` 시 포함됩니다.",
    )



class TranscribeAcceptedResponse(BaseModel):
    """업로드 성공 시 반환. 이 `task_id`로 결과를 폴링합니다."""

    task_id: str = Field(
        ...,
        description="작업 추적용 고유 ID (12자 hex)",
        examples=["a1b2c3d4e5f6"],
        min_length=12,
        max_length=12,
    )
    status: str = Field(
        "pending",
        description="초기 상태 (항상 `pending`)",
        examples=["pending"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"task_id": "a1b2c3d4e5f6", "status": "pending"}],
        },
    }


class UtteranceResult(BaseModel):
    """발화 분리 결과.

    화자 변경 + 묵음 구간을 기준으로 분리된 개별 발화입니다.
    `split_by_utterance=true` 시 각 발화별 WAV 파일이 생성됩니다.
    """

    index: int = Field(
        ...,
        description="발화 인덱스 (0-based)",
        examples=[0],
        ge=0,
    )
    start_sec: float = Field(
        ...,
        description="발화 시작 시간 (초)",
        examples=[0.03],
    )
    end_sec: float = Field(
        ...,
        description="발화 종료 시간 (초)",
        examples=[2.79],
    )
    duration_sec: float = Field(
        ...,
        description="발화 길이 (초)",
        examples=[2.76],
    )
    speaker_id: str = Field(
        ...,
        description="화자 ID (SPEAKER_00, SPEAKER_01, ...)",
        examples=["SPEAKER_00"],
    )
    transcript_text: str = Field(
        ...,
        description="발화 텍스트 (PII 마스킹 적용)",
        examples=["왜 그러지? 어 끊어졌네 갑자기"],
    )
    audio_filename: Optional[str] = Field(
        None,
        description="발화별 WAV 파일명. `split_by_utterance=true` 시 생성. "
        "`GET /api/v1/jobs/{task_id}/audio/{filename}`으로 다운로드",
        examples=["utterance_000.wav"],
    )
    words: Optional[list[WordResponse]] = Field(
        None,
        description="발화 내 단어 단위 타임스탬프 목록",
    )
    emotion: Optional[str] = Field(
        None,
        description="자동 예측 감정 3종 (긍정|중립|부정). 모델 없을 때 null.",
        examples=["긍정"],
    )
    emotion_confidence: Optional[float] = Field(
        None,
        description="감정 예측 신뢰도 0.0–1.0. ≥0.85=auto_confirmed, 0.60–0.85=auto_review, <0.60=needs_review",
        examples=[0.92],
    )
    dialog_act_confidence: Optional[float] = Field(
        None,
        description="대화행위 예측 신뢰도 0.0–1.0",
        examples=[0.78],
    )
    auto_label_model_version: Optional[str] = Field(
        None,
        description="예측에 사용된 모델 버전 (예: v20250513_120000)",
        examples=["v20250513_120000"],
    )


class SpeakerAudioResult(BaseModel):
    """화자별 오디오 결과.

    mute 방식으로 생성 — 원본 타임라인 유지, 상대방 구간은 무음 처리.
    `split_by_speaker=true` 시 생성됩니다.
    """

    speaker_id: str = Field(
        ...,
        description="화자 ID",
        examples=["SPEAKER_01"],
    )
    total_duration_sec: float = Field(
        ...,
        description="화자의 총 발화 길이 (초)",
        examples=[246.91],
    )
    audio_filename: Optional[str] = Field(
        None,
        description="화자별 WAV 파일명. "
        "`GET /api/v1/jobs/{task_id}/audio/{filename}`으로 다운로드",
        examples=["speaker_speaker_01.wav"],
    )


class TranscribeResultResponse(BaseModel):
    """STT 완료 시 반환되는 전체 결과.

    세그먼트별 타임스탬프, PII 마스킹 처리된 텍스트,
    PII 감지 요약을 포함합니다.
    화자분리/발화분리/화자별 오디오 분리 옵션에 따라
    추가 필드가 포함됩니다.
    """

    task_id: str = Field(..., description="작업 ID", examples=["a1b2c3d4e5f6"])
    status: str = Field(
        "completed",
        description="작업 상태 (completed)",
        examples=["completed"],
    )
    language: str = Field(
        ...,
        description="인식된 언어 코드 (ISO 639-1)",
        examples=["ko"],
    )
    duration_seconds: float = Field(
        ...,
        description="원본 오디오 길이 (초)",
        examples=[20.17],
        ge=0,
    )
    processing_seconds: Optional[float] = Field(
        None,
        description="STT 처리 소요 시간 (초)",
        examples=[3.45],
    )
    audio_stats: Optional[dict] = Field(
        None,
        description="오디오 분석 통계 (sample_rate, channels, bitrate, rms, "
        "silence_ratio, snr_db, clipping_ratio, effective_minutes)",
    )
    segments: list[SegmentResponse] = Field(
        ...,
        description="타임스탬프 기반 세그먼트 목록. 각 세그먼트에 start/end/text 포함. "
        "`diarize=true` 시 words 배열 포함",
    )
    full_text: str = Field(
        ...,
        description="전체 텍스트 (모든 세그먼트 텍스트를 공백으로 결합, PII 마스킹 적용)",
        examples=["안녕하세요 김OO입니다 제 번호는 010-****-5678이에요"],
    )
    pii_summary: list[PIIDetectedItem] = Field(
        ...,
        description="PII 유형별 감지 건수 요약. 감지된 PII가 없으면 빈 배열",
    )
    diarization_enabled: bool = Field(
        ...,
        description="화자분리 적용 여부. `diarize=true`로 요청하더라도 "
        "HF_TOKEN 미설정 시 `false`로 반환됩니다.",
    )
    utterances: Optional[list[UtteranceResult]] = Field(
        None,
        description="발화 분리 결과 목록. `diarize=true` + `split_by_utterance=true`일 때만 포함. "
        "화자 변경 + 0.5초 묵음 기준으로 분리되며, 각 발화별 WAV 파일 매핑",
    )
    speaker_audio: Optional[list[SpeakerAudioResult]] = Field(
        None,
        description="화자별 mute 오디오 목록. `diarize=true` + `split_by_speaker=true`일 때만 포함. "
        "원본 타임라인 유지, 상대방 구간은 무음 처리",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task_id": "a1b2c3d4e5f6",
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
                            ],
                        },
                    ],
                    "speaker_audio": [
                        {
                            "speaker_id": "SPEAKER_01",
                            "total_duration_sec": 246.91,
                            "audio_filename": "speaker_speaker_01.wav",
                        },
                    ],
                },
            ],
        },
    }


class JobPendingResponse(BaseModel):
    """작업이 아직 진행 중일 때 반환되는 응답.

    관측용 optional 필드(position_in_queue 등)는 폴링 타임아웃 원인 분석용이며,
    기존 클라이언트는 무시해도 무방하다.
    """

    task_id: str = Field(
        ...,
        description="작업 ID",
        examples=["a1b2c3d4e5f6"],
    )
    status: str = Field(
        ...,
        description="작업 상태 (`pending` 또는 `processing`)",
        examples=["processing"],
    )
    position_in_queue: Optional[int] = Field(
        None,
        description="queued_at 기준 대기 순번 (1-based). GPU 점유 중 포함. 완료 시 null.",
        examples=[2],
    )
    queue_size: Optional[int] = Field(
        None,
        description="현재 pending + processing 상태 작업 수",
        examples=[3],
    )
    gpu_busy: Optional[bool] = Field(
        None,
        description="현재 GPU 세마포어 점유 여부 (gpu_acquired_at 있고 gpu_released_at 없음)",
    )
    queued_at: Optional[float] = Field(
        None,
        description="작업 생성 시각 (unix seconds)",
    )
    gpu_acquired_at: Optional[float] = Field(
        None,
        description="GPU 세마포어 획득 시각 (unix seconds). 획득 전은 null.",
    )
    elapsed_queue_seconds: Optional[float] = Field(
        None,
        description="큐 대기 시간(초). GPU 획득 전은 현재까지 대기, 획득 후는 대기 총 시간.",
        examples=[45.2],
    )
    elapsed_processing_seconds: Optional[float] = Field(
        None,
        description="GPU 처리 시간(초). GPU 획득 후부터 현재(또는 release)까지.",
        examples=[12.8],
    )


class TranscribeResponse(BaseModel):
    task_id: str
    status: str


class JobResultResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """에러 응답 형식."""

    detail: str = Field(
        ...,
        description="에러 메시지",
        examples=["Unsupported format: txt"],
    )


class QueueStatus(BaseModel):
    """큐 백프레셔 상태 정보.

    `POST /api/v1/transcribe`는 `active >= max_active` 조건에서 503을 반환합니다.
    모니터링/대시보드에서 이 값을 폴링하여 큐 포화 상태를 관측할 수 있습니다.
    """

    active: int = Field(
        ...,
        description="현재 pending + processing 상태인 작업 수",
        examples=[3],
        ge=0,
    )
    max_active: int = Field(
        ...,
        description="서버 설정의 최대 동시 작업 수 (`MAX_ACTIVE_JOBS`). "
        "이 값에 도달하면 신규 POST는 503 반환.",
        examples=[5],
        ge=1,
    )
    utilization_pct: float = Field(
        ...,
        description="큐 사용률 (%) = active / max_active * 100. "
        "100 이상은 큐 포화 상태.",
        examples=[60.0],
        ge=0,
    )
    gpu_busy: Optional[bool] = Field(
        None,
        description="현재 GPU 세마포어 점유 여부 (acquired 후 release 전)",
    )
    current_task_id: Optional[str] = Field(
        None,
        description="현재 GPU를 점유 중인 task_id. 없으면 null",
        examples=["a1b2c3d4e5f6"],
    )
    queue_depth: Optional[int] = Field(
        None,
        description="pending + processing 중 GPU 미점유 task 수 (현재 처리 중 제외)",
        examples=[2],
    )
    waiting_task_ids: Optional[list[str]] = Field(
        None,
        description="대기 중인 task_id 목록 (queued_at 오름차순, 현재 GPU 점유 task 제외)",
    )


class HealthResponse(BaseModel):
    """서버 상태 및 GPU/모델 정보."""

    status: str = Field(
        ...,
        description="서버 상태 (`ok`)",
        examples=["ok"],
    )
    service: str = Field(
        ...,
        description="서비스 이름",
        examples=["WhisperX STT Server"],
    )
    version: str = Field(
        ...,
        description="API 버전 (semver)",
        examples=["2.0.0"],
    )
    model: str = Field(
        ...,
        description="WhisperX 모델 크기",
        examples=["large-v3"],
    )
    device: str = Field(
        ...,
        description="추론 디바이스 (`cuda` 또는 `cpu`)",
        examples=["cuda"],
    )
    gpu: Optional[str] = Field(
        None,
        description="NVIDIA GPU 모델명. CUDA 미사용 시 `null`",
        examples=["NVIDIA GeForce RTX 4090"],
    )
    model_loaded: bool = Field(
        ...,
        description="WhisperX 모델이 GPU 메모리에 로딩 완료되었는지 여부. "
        "`false`이면 아직 시작 중이므로 STT 요청을 보내면 안 됩니다.",
    )
    queue: QueueStatus = Field(
        ...,
        description="큐 백프레셔 상태 (active / max_active / utilization_pct)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "service": "WhisperX STT Server",
                    "version": "2.0.0",
                    "model": "large-v3",
                    "device": "cuda",
                    "gpu": "NVIDIA GeForce RTX 4090",
                    "model_loaded": True,
                    "queue": {
                        "active": 3,
                        "max_active": 5,
                        "utilization_pct": 60.0,
                    },
                },
            ],
        },
    }
