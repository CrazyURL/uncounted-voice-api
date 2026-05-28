"""B-60 quality 회귀 guard — d40a8fd cherry-pick 회귀 잠금.

배경:
  d40a8fd 이전 worker.py 의 ``_get_audio_stats_sync`` 는
  ``["ffprobe", "-v", "error", "-af", "astats=...", "-f", "null", "-i", wav_path]`` 호출.
  ffprobe 는 ``-af`` (audio filter graph)를 실행하지 않으므로 + ``-v error`` 가 astats 가
  내는 info-level stderr 라인까지 억제 → stderr 비어있음 → parser 가 "RMS level dB" /
  "Peak level dB" 라인을 못 찾음 → ``rms_db = peak_db = -60.0`` 초기값 유지 → silencedetect
  도 동일 silent 실패 → ``total_silence = 0`` → ``_compute_quality`` 가 모든 입력에 대해
  deterministic 으로 ``snr_db=0 / speech_ratio=1.0 / score=60 / grade="B"`` 산출.

d40a8fd 의 fix 표면 (본 테스트가 lock 하는 결정):
  1. subprocess command 첫 토큰 = ``"ffmpeg"`` (NOT ``"ffprobe"``) — astats / silencedetect 양쪽.
  2. ``-v`` 인자 = ``"info"`` (NOT ``"error"``) — astats stderr 가 parser 에 도달하기 위함.
  3. ``returncode != 0`` 일 때 ``log.warning`` emit — silent fallback observability gap 차단.

본 테스트는 위 3 결정 표면을 직접 잠그고, 정상 stderr 입력에 대한 parsing 회귀가 없음을 확인한다.
실제 ffmpeg 바이너리는 호출하지 않음(``subprocess.run`` 전부 mock) — dev/CI 환경 의존 0.
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# GPU server 미장착 환경(dev PC / CI)에서도 import 가 통과되도록 heavy deps stub.
# conftest.py 가 whisperx 를 stub 하는 패턴과 동형. _get_audio_stats_sync 자체는
# subprocess.run 만 사용하므로 aiohttp/boto3/supabase 가짜로 충분.
sys.modules.setdefault("aiohttp", MagicMock())
sys.modules.setdefault("boto3", MagicMock())
sys.modules.setdefault("botocore", MagicMock())
sys.modules.setdefault("botocore.config", MagicMock())
sys.modules.setdefault("supabase", MagicMock())
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "dummy")
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "dummy")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "dummy")

from app.worker import _get_audio_stats_sync  # noqa: E402


def _fake_proc(stderr: str = "", returncode: int = 0, stdout: str = "1.0"):
    """가짜 CompletedProcess — text=True 모드의 capture_output 결과 형태."""
    m = MagicMock()
    m.stderr = stderr
    m.returncode = returncode
    m.stdout = stdout
    return m


def _route_run(astats_resp, silence_resp, dur_resp=None):
    """subprocess.run 호출을 명령에 따라 라우팅하는 side_effect factory.

    - astats 필터 포함 호출 → ``astats_resp``
    - silencedetect 필터 포함 호출 → ``silence_resp``
    - ffprobe (duration 조회) → ``dur_resp`` (기본: stdout="1.0")
    """
    if dur_resp is None:
        dur_resp = _fake_proc(stdout="1.0")
    captured: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        joined = " ".join(cmd)
        if "astats" in joined:
            return astats_resp
        if "silencedetect" in joined:
            return silence_resp
        # 나머지 ffprobe = duration 조회
        return dur_resp

    return _run, captured


# ─────────────────────────────────────────────────────────────────────────────
# 결정 표면 1: astats / silencedetect 호출은 ffmpeg 으로 (ffprobe -af 회귀 차단)
# ─────────────────────────────────────────────────────────────────────────────

def test_astats_uses_ffmpeg_not_ffprobe():
    """B-60 guard: astats 호출 command[0] == 'ffmpeg' 이어야 한다.

    d40a8fd 회귀(ffprobe 로 되돌림) 시 본 테스트 RED.
    """
    astats = _fake_proc(stderr="  RMS level dB: -20.0\n  Peak level dB: -10.0\n")
    silence = _fake_proc(stderr="")
    run_fn, captured = _route_run(astats, silence)

    with patch("subprocess.run", side_effect=run_fn):
        _get_audio_stats_sync("/tmp/x.wav")

    astats_cmds = [c for c in captured if any("astats" in s for s in c)]
    assert len(astats_cmds) == 1, f"astats 호출 1회 기대, 실제 {len(astats_cmds)} 회"
    assert astats_cmds[0][0] == "ffmpeg", (
        f"astats 호출이 {astats_cmds[0][0]!r} 으로 시작함 — ffprobe 면 B-60 회귀"
    )


def test_silencedetect_uses_ffmpeg_not_ffprobe():
    """B-60 guard: silencedetect 호출 command[0] == 'ffmpeg' 이어야 한다."""
    astats = _fake_proc(stderr="  RMS level dB: -20.0\n  Peak level dB: -10.0\n")
    silence = _fake_proc(stderr="")
    run_fn, captured = _route_run(astats, silence)

    with patch("subprocess.run", side_effect=run_fn):
        _get_audio_stats_sync("/tmp/x.wav")

    silence_cmds = [c for c in captured if any("silencedetect" in s for s in c)]
    assert len(silence_cmds) == 1, f"silencedetect 호출 1회 기대, 실제 {len(silence_cmds)} 회"
    assert silence_cmds[0][0] == "ffmpeg", (
        f"silencedetect 호출이 {silence_cmds[0][0]!r} 으로 시작함 — ffprobe 면 B-60 회귀"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 결정 표면 2: -v info (NOT -v error) — astats stderr 가 parser 에 도달해야
# ─────────────────────────────────────────────────────────────────────────────

def test_ffmpeg_v_info_not_v_error():
    """B-60 guard: ffmpeg 호출의 -v 인자 = 'info'.

    -v error 면 astats 가 info-level 로 emit 하는 RMS/Peak 라인이 stderr 에서 사라져
    parser 가 -60 fallback 으로 떨어진다.
    """
    astats = _fake_proc(stderr="  RMS level dB: -20.0\n  Peak level dB: -10.0\n")
    silence = _fake_proc(stderr="")
    run_fn, captured = _route_run(astats, silence)

    with patch("subprocess.run", side_effect=run_fn):
        _get_audio_stats_sync("/tmp/x.wav")

    ffmpeg_cmds = [c for c in captured if c[0] == "ffmpeg"]
    assert len(ffmpeg_cmds) >= 2, f"ffmpeg 호출 ≥2 기대 (astats + silencedetect), 실제 {len(ffmpeg_cmds)}"
    for cmd in ffmpeg_cmds:
        # -v 다음 토큰 추출
        try:
            v_idx = cmd.index("-v")
        except ValueError:
            pytest.fail(f"ffmpeg 호출에 -v 인자 자체가 없음: {cmd}")
        v_level = cmd[v_idx + 1]
        assert v_level == "info", (
            f"ffmpeg 호출 -v 인자={v_level!r} — 'info' 기대 ('error' 면 stderr 억제, B-60 회귀)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 결정 표면 3: returncode != 0 → log.warning emit (silent fallback observability)
# ─────────────────────────────────────────────────────────────────────────────

def test_returncode_failure_emits_log_warning_astats(caplog):
    """B-60 guard: ffmpeg astats 실패 시 log.warning('ffmpeg astats failed ...')."""
    astats = _fake_proc(stderr="some ffmpeg error", returncode=1)
    silence = _fake_proc(stderr="")
    run_fn, _ = _route_run(astats, silence)

    with caplog.at_level(logging.WARNING, logger="gpu_worker"):
        with patch("subprocess.run", side_effect=run_fn):
            _get_audio_stats_sync("/tmp/x.wav")

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("ffmpeg astats failed" in m for m in warnings), (
        f"astats 실패 경고 누락 — silent fallback observability gap (B-60 회귀 signal). 캡처={warnings}"
    )


def test_returncode_failure_emits_log_warning_silencedetect(caplog):
    """B-60 guard: ffmpeg silencedetect 실패 시 log.warning('ffmpeg silencedetect failed ...')."""
    astats = _fake_proc(stderr="  RMS level dB: -20.0\n  Peak level dB: -10.0\n")
    silence = _fake_proc(stderr="some ffmpeg error", returncode=2)
    run_fn, _ = _route_run(astats, silence)

    with caplog.at_level(logging.WARNING, logger="gpu_worker"):
        with patch("subprocess.run", side_effect=run_fn):
            _get_audio_stats_sync("/tmp/x.wav")

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("ffmpeg silencedetect failed" in m for m in warnings), (
        f"silencedetect 실패 경고 누락 — silent fallback observability gap. 캡처={warnings}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 회귀 0: 정상 stderr 입력 → RMS/Peak 파싱이 동작한다 (fix 후에도 parser 살아있음)
# ─────────────────────────────────────────────────────────────────────────────

def test_parses_rms_and_peak_from_ffmpeg_astats_stderr():
    """fix 적용 후에도 RMS/Peak 라인 파싱이 동작해야 한다 (parser 회귀 0)."""
    astats = _fake_proc(
        stderr=(
            "[Parsed_astats_0 @ 0x55b] Channel: 1\n"
            "[Parsed_astats_0 @ 0x55b] RMS level dB: -18.34\n"
            "[Parsed_astats_0 @ 0x55b] Peak level dB: -3.21\n"
            "[Parsed_astats_0 @ 0x55b] Min level: -0.6789\n"
        )
    )
    silence = _fake_proc(stderr="")  # 무음 없음
    run_fn, _ = _route_run(astats, silence, dur_resp=_fake_proc(stdout="5.0"))

    with patch("subprocess.run", side_effect=run_fn):
        stats = _get_audio_stats_sync("/tmp/x.wav")

    assert stats["rms_db"] == -18.34, f"RMS 파싱 실패: {stats}"
    assert stats["peak_db"] == -3.21, f"Peak 파싱 실패: {stats}"
    assert stats["silence_ratio"] == 0.0


def test_parses_silence_ratio_from_silencedetect_stderr():
    """fix 적용 후에도 silencedetect 파싱이 동작해야 한다 (parser 회귀 0)."""
    astats = _fake_proc(stderr="  RMS level dB: -20.0\n  Peak level dB: -10.0\n")
    silence = _fake_proc(
        stderr=(
            "[silencedetect @ 0x55c] silence_start: 1.0\n"
            "[silencedetect @ 0x55c] silence_end: 2.5 | silence_duration: 1.5\n"
        )
    )
    run_fn, _ = _route_run(astats, silence, dur_resp=_fake_proc(stdout="10.0"))

    with patch("subprocess.run", side_effect=run_fn):
        stats = _get_audio_stats_sync("/tmp/x.wav")

    # 1.5초 무음 / 10초 = 0.15
    assert abs(stats["silence_ratio"] - 0.15) < 1e-6, f"silence_ratio 파싱 실패: {stats}"


# ─────────────────────────────────────────────────────────────────────────────
# B-60 RED 시그니처 lock: stderr 비어있으면 -60/-60 fallback (의도된 contract)
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_to_minus_60_when_stderr_lacks_rms_peak():
    """B-60 시그니처 명시: stderr 에 RMS/Peak 라인이 없으면 rms=peak=-60 fallback.

    이는 fix 전후 모두 동일한 동작이지만, 본 테스트가 그 contract 를 lock 한다 —
    Phase 4 의 GREEN 비교 baseline (운영에서 같은 결과가 나오면 = ffmpeg 가 stderr 안 내고
    있다는 signal, returncode warning + 별도 진단으로 추적).
    """
    astats = _fake_proc(stderr="")  # info 라인 없음 — 비정상 상태
    silence = _fake_proc(stderr="")
    run_fn, _ = _route_run(astats, silence)

    with patch("subprocess.run", side_effect=run_fn):
        stats = _get_audio_stats_sync("/tmp/x.wav")

    assert stats["rms_db"] == -60.0, f"stderr 비어있을 때 fallback 깨짐: {stats}"
    assert stats["peak_db"] == -60.0, f"stderr 비어있을 때 fallback 깨짐: {stats}"
