"""orphan cleanup env helper 회귀 테스트.

검증 대상 (advisor 권고: 결정 표면 minimal):
  - 미설정 → 기본값 (False / True / 0.5).
  - 명시 truthy/falsy 문자열 → 정확한 bool / float.
  - invalid → ValueError fail-loud (WORKER_CONCURRENCY 컨벤션 일치).

stdlib float() / case-insensitive 문자열 매핑 그 자체는 별도 테스트하지 않음.
GPU / aiohttp / supabase 의존 없음 — env helper 만 import.
"""

from __future__ import annotations

import pytest

from app.worker_config import (
    resolve_orphan_cleanup_dry_run,
    resolve_orphan_cleanup_enabled,
    resolve_orphan_cleanup_min_ratio,
)


# ─────────────────────────────────────────────────────────────────────────────
# Default values (미설정 = 기본)
# ─────────────────────────────────────────────────────────────────────────────

def test_enabled_default_false():
    assert resolve_orphan_cleanup_enabled({}) is False


def test_dry_run_default_true():
    assert resolve_orphan_cleanup_dry_run({}) is True


def test_min_ratio_default_0_5():
    assert resolve_orphan_cleanup_min_ratio({}) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Truthy / falsy parsing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["true", "True", "TRUE", "1", "yes", "on"])
def test_enabled_truthy_variants(raw):
    assert resolve_orphan_cleanup_enabled({"WORKER_ORPHAN_CLEANUP_ENABLED": raw}) is True


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off"])
def test_enabled_falsy_variants(raw):
    assert resolve_orphan_cleanup_enabled({"WORKER_ORPHAN_CLEANUP_ENABLED": raw}) is False


def test_dry_run_explicit_false():
    assert resolve_orphan_cleanup_dry_run({"WORKER_ORPHAN_CLEANUP_DRY_RUN": "false"}) is False


def test_dry_run_explicit_true():
    assert resolve_orphan_cleanup_dry_run({"WORKER_ORPHAN_CLEANUP_DRY_RUN": "true"}) is True


def test_min_ratio_valid_float():
    assert resolve_orphan_cleanup_min_ratio(
        {"WORKER_ORPHAN_CLEANUP_MIN_RATIO": "0.75"}
    ) == 0.75


def test_min_ratio_zero_accepted():
    # 0.0 = 가드 disable, operator 의 명시 선택.
    assert resolve_orphan_cleanup_min_ratio(
        {"WORKER_ORPHAN_CLEANUP_MIN_RATIO": "0.0"}
    ) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Invalid → ValueError (fail-loud)
# ─────────────────────────────────────────────────────────────────────────────

def test_enabled_invalid_raises():
    with pytest.raises(ValueError):
        resolve_orphan_cleanup_enabled({"WORKER_ORPHAN_CLEANUP_ENABLED": "maybe"})


def test_dry_run_invalid_raises():
    with pytest.raises(ValueError):
        resolve_orphan_cleanup_dry_run({"WORKER_ORPHAN_CLEANUP_DRY_RUN": "dunno"})


def test_min_ratio_invalid_raises():
    with pytest.raises(ValueError):
        resolve_orphan_cleanup_min_ratio({"WORKER_ORPHAN_CLEANUP_MIN_RATIO": "abc"})
