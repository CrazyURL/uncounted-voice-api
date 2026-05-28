"""``VOICE_API_MAX_WAIT_SEC`` env 연동 회귀 테스트.

검증 대상 (advisor 권고: 3 가지만):
  - 미설정 → 300 (기본값이 실제로 살아 있음 — hardcoded 300 회귀 가드).
  - 유효 정수 문자열 → 지정값 (상수가 정말 env-fed 인지 확인 — hardcoded 300 회귀 가드).
  - invalid 문자열 → ValueError (fail-loud semantics 확인 — 명시된 설계 결정).

stdlib ``int()`` semantics(0/음수/소수점/exponential 등)는 Python 자체 동작이라
별도 테스트 안 함 — 위 3가지가 본 PR 의 모든 결정 표면을 커버.

GPU/모델/whisperx/aiohttp 불요 — env 해석 helper 만 import.
"""

from __future__ import annotations

import pytest

from app.worker_config import resolve_voice_api_max_wait_sec


def test_unset_returns_default_300():
    # 빈 env 주입 → 기본값 300 (현 하드코딩 동작 보존, 회귀 가드).
    assert resolve_voice_api_max_wait_sec({}) == 300


def test_valid_int_env_used():
    # 장통화 시나리오: 7200초(2시간) 설정 → 그대로 사용.
    assert resolve_voice_api_max_wait_sec({"VOICE_API_MAX_WAIT_SEC": "7200"}) == 7200
    # 짧게 줄이는 케이스도 동작.
    assert resolve_voice_api_max_wait_sec({"VOICE_API_MAX_WAIT_SEC": "60"}) == 60


def test_invalid_env_raises_value_error():
    # fail-loud: 비-정수 문자열은 import 시점에 ValueError 로 즉시 실패.
    # WORKER_CONCURRENCY 의 ``int(os.getenv(..., "2"))`` 와 동일한 컨벤션.
    with pytest.raises(ValueError):
        resolve_voice_api_max_wait_sec({"VOICE_API_MAX_WAIT_SEC": "abc"})
