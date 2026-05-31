import os
from pathlib import Path

# Environment
ENV = os.environ.get("ENV", "dev")
PORT = int(os.environ.get("PORT", "8001" if ENV == "dev" else "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")
WORKERS = int(os.environ.get("WORKERS", "1"))

# WhisperX 모델 설정 — STT Pipeline preset 기반
# v2-largev3-int8 (default) : large-v3 + int8 + batch_size=4 (RTX 4060 8GB 안전 상한)
# v1-turbo-frozen (legacy)  : large-v3-turbo + float16 + batch_size=4
# MODEL_SIZE / COMPUTE_TYPE / BATCH_SIZE env 명시 시 그 값이 우선 (운영자 override 존중).
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "v2-largev3-int8")

_PIPELINE_PRESETS = {
    "v2-largev3-int8": {
        "model_size": "large-v3",
        "compute_type": "int8",
        "batch_size": 4,
    },
    "v1-turbo-frozen": {
        "model_size": "large-v3-turbo",
        "compute_type": "float16",
        "batch_size": 4,
    },
}
_preset = _PIPELINE_PRESETS.get(PIPELINE_VERSION, _PIPELINE_PRESETS["v2-largev3-int8"])

MODEL_SIZE = os.environ.get("MODEL_SIZE", _preset["model_size"])
DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", _preset["compute_type"])
LANGUAGE = os.environ.get("LANGUAGE", "ko")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", str(_preset["batch_size"])))

# OOM 가드 — _model.transcribe OOM 시 batch_size 절반씩 후퇴 (예: 4 → 2 → 1)
BATCH_SIZE_MIN = int(os.environ.get("BATCH_SIZE_MIN", "1"))
BATCH_OOM_RETRY_ENABLED = os.environ.get("BATCH_OOM_RETRY_ENABLED", "true").lower() in ("true", "1", "yes")

# HuggingFace 토큰 (화자분리용)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# 파일 경로 (RAM 디스크)
TEMP_DIR = Path(os.environ.get("TEMP_DIR", "/dev/shm/stt-temp"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/dev/shm/stt-results"))

# 업로드 제한
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", str(500 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4", "amr", "3gp"}

# 큐 백프레셔: pending + processing 합산이 이 값 이상이면 POST /transcribe는 503 반환
MAX_ACTIVE_JOBS = int(os.environ.get("MAX_ACTIVE_JOBS", "5"))
QUEUE_FULL_RETRY_AFTER_SEC = int(os.environ.get("QUEUE_FULL_RETRY_AFTER_SEC", "30"))
# processing/pending 상태가 이 시간을 초과하면 stuck으로 간주하고 failed 처리
MAX_PROCESSING_AGE_SEC = int(os.environ.get("MAX_PROCESSING_AGE_SEC", "1800"))  # 30분

# 발화 분리 (Utterance Segmentation)
SILENCE_GAP_SEC = float(os.environ.get("SILENCE_GAP_SEC", "0.5"))
MIN_UTTERANCE_SEC = float(os.environ.get("MIN_UTTERANCE_SEC", "5.0"))
MAX_UTTERANCE_SEC = float(os.environ.get("MAX_UTTERANCE_SEC", "30.0"))
SHORT_ANSWER_MIN_SEC = float(os.environ.get("SHORT_ANSWER_MIN_SEC", "0.3"))
PADDING_SEC = float(os.environ.get("PADDING_SEC", "0.15"))

# PII 마스킹
PII_MASK_PAD_SEC = float(os.environ.get("PII_MASK_PAD_SEC", "0.15"))

# 로깅
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG" if ENV == "dev" else "INFO")

# 서버 정보
VERSION = "2.0.0"
SERVICE_NAME = "WhisperX STT Server"

# 발화 분리 상수 (UtteranceSegmenter)
SILENCE_GAP_SEC = 0.5
MIN_UTTERANCE_SEC = 5.0
MAX_UTTERANCE_SEC = 30.0
SHORT_ANSWER_MIN_SEC = 0.3
PADDING_SEC = 0.15
SHORT_ANSWER_WORDS = [
    # 긍정 응답
    "네", "넵", "넹",
    "예", "옙",
    "응", "응응", "엉",
    # 부정 응답
    "아니", "아니요", "아뇨", "아니야", "아니에요",
    # 동의/인정
    "그래", "그래요", "그럼", "그럼요", "그렇죠", "그렇지",
    "맞아", "맞아요", "맞네", "맞죠",
    "좋아", "좋아요", "좋죠",
    "알겠어", "알겠어요", "알았어", "알았어요",
    "오케이", "오케", "OK",
    # 망설임 (선별적)
    "음", "흠",
]
SAMPLE_RATE = 16000

# Audio Preprocessing — 보수적 임계값 (품질 보존 우선)
SILENCE_RMS_THRESHOLD = float(os.environ.get("SILENCE_RMS_THRESHOLD", "0.005"))
DUPLICATE_WINDOW_SEC = float(os.environ.get("DUPLICATE_WINDOW_SEC", "2.5"))
DUPLICATE_CORR_THRESHOLD = float(os.environ.get("DUPLICATE_CORR_THRESHOLD", "0.95"))
PREPROCESS_FRAME_MS = int(os.environ.get("PREPROCESS_FRAME_MS", "20"))

# 무음 압축 전용 임계값 (SILENCE_GAP_SEC 발화분리용과 분리)
SILENCE_COMPRESS_MIN_SEC = float(os.environ.get("SILENCE_COMPRESS_MIN_SEC", "1.0"))
SILENCE_COMPRESS_TARGET_SEC = float(os.environ.get("SILENCE_COMPRESS_TARGET_SEC", "0.5"))

# denoise 후 silence_compress가 사용할 동적 임계값 (Round 3 진단 실측 p50=0.00090 기준)
# DeepFilterNet이 voice RMS를 median 23배 감쇠시키므로 기본 0.005 threshold가 cascade 손실을 유발.
# 0.0005로 낮추어 감쇠된 voice frame이 silence로 오분류되지 않게 한다.
SILENCE_RMS_THRESHOLD_DENOISE = float(os.environ.get("SILENCE_RMS_THRESHOLD_DENOISE", "0.0005"))

# 발화 끝 떠돌이 단어(hanging word) 보정 — 직전 단어와 이 간격 이상이면 다음 발화로 이동
HANGING_WORD_GAP_SEC = float(os.environ.get("HANGING_WORD_GAP_SEC", "0.3"))

# Gain Normalize 최대 증폭 (노이즈 증폭 방지)
MAX_GAIN_X = float(os.environ.get("MAX_GAIN_X", "10.0"))
# 로컬 게인 정규화 최대 증폭 — 글로벌보다 높게 허용하여 조용한 구간의 VAD 감지 개선
LOCAL_MAX_GAIN_X = float(os.environ.get("LOCAL_MAX_GAIN_X", "30.0"))

# WhisperX 내부 silero VAD 임계값 (낮을수록 조용한 speech 감지 향상)
# 기본값: onset=0.500, offset=0.363 — 조용한 구간 누락 시 낮춤
# 0.15로 낮춰야 작은 음량(volume < -20dBFS) 구간도 speech로 인식
VAD_ONSET = float(os.environ.get("VAD_ONSET", "0.150"))
VAD_OFFSET = float(os.environ.get("VAD_OFFSET", "0.100"))

# STT 힌트 (고유명사 인식 개선)
HOTWORDS = os.environ.get("HOTWORDS", None)
INITIAL_PROMPT = os.environ.get("INITIAL_PROMPT", None)

# ─────────────────────────────────────────────────────────────
# 전처리 파이프라인 단계별 토글 (품질 보존 점진 활성화)
# Round 1: gain만 ON → Round 2: + silence → Round 3: + denoise → Round 4: + dedup
# ─────────────────────────────────────────────────────────────
PREPROCESS_GAIN_ENABLED = os.environ.get("PREPROCESS_GAIN_ENABLED", "true").lower() in ("true", "1", "yes")
PREPROCESS_DENOISE_ENABLED = os.environ.get("PREPROCESS_DENOISE_ENABLED", "false").lower() in ("true", "1", "yes")
PREPROCESS_DEDUP_ENABLED = os.environ.get("PREPROCESS_DEDUP_ENABLED", "false").lower() in ("true", "1", "yes")
PREPROCESS_SILENCE_ENABLED = os.environ.get("PREPROCESS_SILENCE_ENABLED", "false").lower() in ("true", "1", "yes")

# 레거시 호환 (deprecated — 제거 예정)
DENOISE_ENABLED = PREPROCESS_DENOISE_ENABLED

# Deduplication: 슬라이딩 윈도우 최대 룩어헤드 (5 → 3, 오탐 감소)
MAX_DEDUP_LOOKAHEAD = int(os.environ.get("MAX_DEDUP_LOOKAHEAD", "3"))

# 대용량 오디오 청크 분할
CHUNK_DURATION_SEC = int(os.environ.get("CHUNK_DURATION_SEC", "1800"))    # 목표 청크 길이 (30분)
CHUNK_THRESHOLD_SEC = int(os.environ.get("CHUNK_THRESHOLD_SEC", "3600"))  # 이 길이 이상만 분할 (1시간)
CHUNK_SILENCE_DB = float(os.environ.get("CHUNK_SILENCE_DB", "-30"))       # 무음 감지 임계값 (dB)
CHUNK_SILENCE_DUR = float(os.environ.get("CHUNK_SILENCE_DUR", "0.3"))     # 최소 무음 길이 (초)
CHUNK_MARGIN_SEC = int(os.environ.get("CHUNK_MARGIN_SEC", "300"))         # 분할 지점 탐색 범위 (±5분)
