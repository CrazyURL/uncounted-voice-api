"""
Training router — 감정 모델 재학습 트리거 및 상태 조회

POST /api/v1/training/start   → subprocess 로 train_emotion_model.py 실행, job_id 반환
GET  /api/v1/training/status/{job_id} → training_status.json 폴링
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["training"])

MODEL_BASE_DIR = Path(os.environ.get("EMOTION_MODEL_DIR", "models/emotion"))
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

# 실행 중인 job 추적 {job_id: status_file_path}
_running_jobs: dict[str, Path] = {}


class TrainingStartRequest(BaseModel):
    base_model_path: Optional[str] = None
    previous_model_path: Optional[str] = None
    dummy: bool = False


class TrainingStartResponse(BaseModel):
    job_id: str
    message: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str  # running | completed | failed
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    val_loss: Optional[float] = None
    val_emotion_acc: Optional[float] = None
    model_version: Optional[str] = None
    error: Optional[str] = None
    progress_pct: Optional[float] = None


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(req: TrainingStartRequest):
    """감정 모델 재학습을 백그라운드 subprocess 로 시작한다."""
    job_id = str(uuid.uuid4())

    MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    status_file = MODEL_BASE_DIR / f"job_{job_id}_status.json"

    # 초기 상태 파일 생성
    _write_status(status_file, {"status": "running", "job_id": job_id})

    train_script = SCRIPTS_DIR / "train_emotion_model.py"
    if not train_script.exists():
        raise HTTPException(status_code=500, detail="train_emotion_model.py 스크립트를 찾을 수 없습니다")

    cmd = [sys.executable, str(train_script), "--job-id", job_id, "--status-file", str(status_file)]

    if req.base_model_path:
        cmd += ["--base-model-path", req.base_model_path]
    if req.previous_model_path:
        cmd += ["--previous-model-path", req.previous_model_path]
    if req.dummy:
        cmd.append("--dummy")

    env = os.environ.copy()
    env["EMOTION_MODEL_DIR"] = str(MODEL_BASE_DIR)

    try:
        # 웹 서버와 독립적으로 실행 — 서버 재시작 시에도 학습 프로세스 유지
        subprocess.Popen(
            cmd,
            env=env,
            cwd=str(SCRIPTS_DIR.parent.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _running_jobs[job_id] = status_file
        logger.info("학습 시작: job_id=%s", job_id)
    except Exception as exc:
        _write_status(status_file, {"status": "failed", "job_id": job_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"학습 프로세스 시작 실패: {exc}") from exc

    return TrainingStartResponse(job_id=job_id, message="학습이 시작되었습니다")


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """job_id 에 해당하는 학습 상태를 반환한다."""
    # 메모리 캐시에서 파일 경로 조회
    status_file = _running_jobs.get(job_id)

    # 메모리에 없으면 디스크에서 검색 (재시작 후 재접속 케이스)
    if status_file is None:
        candidate = MODEL_BASE_DIR / f"job_{job_id}_status.json"
        if candidate.exists():
            status_file = candidate
            _running_jobs[job_id] = status_file

    if status_file is None or not status_file.exists():
        raise HTTPException(status_code=404, detail="해당 job_id 를 찾을 수 없습니다")

    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="상태 파일 읽기 실패")

    return TrainingStatusResponse(
        job_id=job_id,
        status=data.get("status", "unknown"),
        current_epoch=data.get("current_epoch"),
        total_epochs=data.get("total_epochs"),
        val_loss=data.get("val_loss"),
        val_emotion_acc=data.get("val_emotion_acc"),
        model_version=data.get("model_version"),
        error=data.get("error"),
        progress_pct=data.get("progress_pct"),
    )


def _write_status(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
