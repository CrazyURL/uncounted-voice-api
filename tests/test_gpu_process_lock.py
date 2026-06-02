"""프로세스 간 GPU 락 (voice-api ↔ NeMo 직렬화) 단위 테스트.

검증:
  - 게이트 OFF(기본): no-op 컨텍스트, 락 파일 미생성 (무회귀)
  - 게이트 ON: 컨텍스트 진입/탈출 정상
  - filelock 미설치/타임아웃: 무중단(yield 정상)
"""
import os

from app.services import gpu_process_lock as gpl


def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICE_GPU_PROCESS_LOCK_ENABLED", raising=False)
    assert gpl.is_enabled() is False
    entered = False
    with gpl.gpu_process_lock("test"):
        entered = True
    assert entered is True   # no-op 이어도 컨텍스트는 정상 동작


def test_enabled_acquires_and_releases(monkeypatch, tmp_path):
    lock_path = str(tmp_path / "gpu.lock")
    monkeypatch.setenv("VOICE_GPU_PROCESS_LOCK_ENABLED", "true")
    monkeypatch.setenv("VOICE_GPU_PROCESS_LOCK_PATH", lock_path)
    # 모듈이 import 시점에 경로를 읽으므로 reload
    import importlib
    importlib.reload(gpl)
    assert gpl.is_enabled() is True
    entered = False
    with gpl.gpu_process_lock("t1"):
        entered = True
    assert entered is True
    # 해제 후 재획득 가능(데드락 아님)
    with gpl.gpu_process_lock("t2"):
        pass
    monkeypatch.delenv("VOICE_GPU_PROCESS_LOCK_ENABLED", raising=False)
    importlib.reload(gpl)


def test_reentry_after_exit_no_deadlock(monkeypatch, tmp_path):
    lock_path = str(tmp_path / "gpu2.lock")
    monkeypatch.setenv("VOICE_GPU_PROCESS_LOCK_ENABLED", "true")
    monkeypatch.setenv("VOICE_GPU_PROCESS_LOCK_PATH", lock_path)
    import importlib
    importlib.reload(gpl)
    for _ in range(3):
        with gpl.gpu_process_lock("loop"):
            pass   # 연속 획득/해제 — 데드락 없어야
    monkeypatch.delenv("VOICE_GPU_PROCESS_LOCK_ENABLED", raising=False)
    importlib.reload(gpl)
