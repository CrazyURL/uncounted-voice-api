import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routers import health, transcribe, training
from app.services.whisperx_service import whisperx_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 테스트 환경(`TESTING=1`)에서는 WhisperX/pyannote 모델 로딩을 건너뛴다.
    # 이유: 실제 서버가 이미 GPU를 점유 중인 상태에서 TestClient가 lifespan을
    # 트리거하면 CUDA OOM이 발생한다. 테스트는 엔드포인트 로직만 검증하므로
    # 모델 없이 동작한다.
    if os.environ.get("TESTING") == "1":
        logger.info("TESTING=1 — 모델 로딩 스킵")
    else:
        whisperx_service.load_models()
        logger.info("=" * 50)
        logger.info("모델 로딩 완료 - 요청 대기 중")
        logger.info("=" * 50)
    yield
    logger.info("서버 종료")


TAGS_METADATA = [
    {
        "name": "transcribe",
        "description": "음성 파일 업로드 및 STT 결과 조회",
    },
    {
        "name": "health",
        "description": "서버 상태 및 GPU/모델 정보 확인",
    },
]

app = FastAPI(
    title=config.SERVICE_NAME,
    version=config.VERSION,
    description=(
        "WhisperX 기반 GPU STT 서버.\n\n"
        "음성 파일을 업로드하면 비동기로 음성인식을 수행하고, "
        "한국어 PII(주민등록번호, 전화번호, 이메일 등)를 자동 마스킹하여 "
        "타임스탬프 기반 세그먼트 결과를 반환합니다.\n\n"
        "## 처리 파이프라인\n\n"
        "```\n"
        "업로드 → WhisperX Transcribe → Forced Alignment → (화자분리) → PII 마스킹 → 결과 저장\n"
        "```\n\n"
        "## 주요 기능\n\n"
        "| 기능 | 설명 |\n"
        "|------|------|\n"
        "| **WhisperX large-v3** | GPU 가속 음성인식 (float16) |\n"
        "| **Forced Alignment** | 단어 단위 정밀 타임스탬프 |\n"
        "| **화자분리** | HF_TOKEN 설정 시 다화자 분리 (pyannote) |\n"
        "| **PII 마스킹 (9종)** | 주민등록번호, 운전면허, 여권, 카드, 이메일, 전화번호, 계좌, IP |\n"
        "| **이름 마스킹** | 한국어 성씨 기반 이름 탐지 (선택) |\n\n"
        "## 사용 흐름\n\n"
        "1. `POST /api/v1/transcribe` — 음성 파일 업로드, `task_id` 수령\n"
        "2. `GET /api/v1/jobs/{task_id}` — 폴링하여 결과 조회\n"
        "3. `completed` 상태가 되면 세그먼트·PII 요약 포함 결과 반환\n\n"
        "## 제약 사항\n\n"
        "- 단일 워커(`--workers 1`) — GPU 모델이 전역 싱글턴\n"
        "- 인메모리 작업 저장소 — 서버 재시작 시 작업 상태 소실\n"
        "- NVIDIA GPU + CUDA 필수 (프로덕션)\n"
        "- `ffmpeg` 시스템 의존성 필요\n"
    ),
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Uncounted Voice API",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(transcribe.router)
app.include_router(training.router)
