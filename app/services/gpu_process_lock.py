"""프로세스 간 GPU 락 — voice-api(STT)와 NeMo 서비스가 GPU 추론을 직렬화.

배경(2026-06-02 측정): voice-api(large-v3 상주 4.8GB) + NeMo 추론(+1.8GB)이
**동시**에 일어나면 VRAM peak ~7.5GB(임계 7.6GB 근접). NeMo 는 별도 프로세스(다른
venv)라 voice-api 의 in-process `_gpu_lock`(threading.Semaphore)을 공유할 수 없다.
파일 기반 프로세스 간 락으로 두 프로세스가 GPU 를 동시에 쓰지 않도록 직렬화한다.

안전:
  - env gate VOICE_GPU_PROCESS_LOCK_ENABLED (기본 false) → 컨텍스트가 no-op
    (기존 동작과 byte-identical, 핫패스 무영향).
  - 락 획득 타임아웃 시 그냥 진행(무중단 — 직렬화는 best-effort, STT 를 막지 않음).
  - filelock 미설치/오류 시도 no-op.
"""
from __future__ import annotations

import contextlib
import logging
import os

logger = logging.getLogger(__name__)

_LOCK_PATH = os.environ.get("VOICE_GPU_PROCESS_LOCK_PATH", "/tmp/voice_gpu.lock")


def is_enabled() -> bool:
    return os.environ.get("VOICE_GPU_PROCESS_LOCK_ENABLED", "false").strip().lower() == "true"


def _timeout() -> float:
    try:
        return float(os.environ.get("VOICE_GPU_PROCESS_LOCK_TIMEOUT", "120"))
    except ValueError:
        return 120.0


@contextlib.contextmanager
def gpu_process_lock(label: str = ""):
    """프로세스 간 GPU 락 컨텍스트.

    게이트 OFF / filelock 미가용 / 타임아웃 → 락 없이 진행(no-op, 무중단).
    """
    if not is_enabled():
        yield
        return
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        yield
        return

    lock = FileLock(_LOCK_PATH, timeout=_timeout())
    acquired = False
    try:
        try:
            lock.acquire()
            acquired = True
            logger.debug("[gpu_lock] 획득 %s", label)
        except Timeout:
            logger.warning("[gpu_lock] 타임아웃 — 락 없이 진행 %s", label)
        yield
    finally:
        if acquired:
            try:
                lock.release()
                logger.debug("[gpu_lock] 해제 %s", label)
            except Exception:  # noqa: BLE001
                pass
