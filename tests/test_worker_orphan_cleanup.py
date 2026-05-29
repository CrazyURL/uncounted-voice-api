"""orphan cleanup 회귀 테스트 — `_cleanup_stale_orphans` 결정 표면 잠금.

검증 시나리오 (디렉터 spec 11항 + 추가 가드 2건):
  1.  env unset → cleanup 호출 진입 자체 안 함 (persist_results 경유).
  2.  dry_run=True → DELETE 호출 0.
  3.  actual (dry_run=False) → safe 행만 DELETE.
  4.  pii_reviewed_at IS NOT NULL → 보존.
  5.  pii_masked_at IS NOT NULL → 보존.
  6.  quality_reviewed_by IS NOT NULL → 보존.
  7.  utterance_human_labels 라벨 존재 → 보존(FK guard).
  8.  new_count=0 → skip(silent-empty-done 가드).
  9.  ratio < min_ratio → skip + warning.
  10. cleanup 실패가 persist_results 전체 실패로 전파되지 않음(fail-closed).
  11. updated_at >= run_start_iso → 본 turn 신규 갱신, 보존.
  12. (추가) utterance_human_labels probe 실패 → 보수적 skip.
  13. (추가) sequence_order > new_count 후보 0 → noop (정상).

GPU/aiohttp/supabase 의존 stub + dummy env 으로 dev PC 단위 검증 가능.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# heavy deps stub (sanitize_json / b60 / max_wait 테스트 동형)
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

from app import worker as worker_module  # noqa: E402
from app.worker import _cleanup_stale_orphans  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fake supabase chain — builder pattern 모킹
# ─────────────────────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """단순 fluent-builder 모킹.

    table(...).select(...).eq(...).gt(...).execute() 호출 시퀀스를 기록하고,
    table 이름과 마지막 호출에 따라 미리 등록된 응답을 반환한다.
    """

    def __init__(self, table_name: str, store: dict):
        self.table_name = table_name
        self.store = store
        self.calls: list[tuple] = []

    def _record(self, method: str, *args):
        self.calls.append((method, args))
        return self

    def select(self, *a, **kw): return self._record("select", a, kw)
    def eq(self, *a, **kw):     return self._record("eq", a, kw)
    def gt(self, *a, **kw):     return self._record("gt", a, kw)
    def in_(self, *a, **kw):    return self._record("in_", a, kw)
    def delete(self, *a, **kw): return self._record("delete", a, kw)

    def execute(self):
        # store[table_name] = {"select": data, "delete": data}
        per_table = self.store.get(self.table_name, {})
        if any(m == "delete" for m, _ in self.calls):
            data = per_table.get("delete", [])
            # 실제 supabase 도 호출이력을 사후 점검 가능하도록 마지막 query 보관.
            self.store.setdefault("_last_delete", {})[self.table_name] = self.calls
            return _FakeResult(data)
        self.store.setdefault("_last_select", {})[self.table_name] = self.calls
        return _FakeResult(per_table.get("select", []))


class _FakeSupabase:
    def __init__(self, store: dict):
        self.store = store

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(name, self.store)


@pytest.fixture
def fake_supabase():
    """worker._supabase + worker._run 을 동기 fake 로 패치."""
    store: dict = {}
    fake = _FakeSupabase(store)
    # _run 은 원본 worker.py 에서 thread executor 로 위임 — 테스트에서는 그냥 lambda 호출.
    async def _fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    with patch.object(worker_module, "_supabase", fake), \
         patch.object(worker_module, "_run", _fake_run):
        yield store


def _row(
    rid: str,
    seq: int,
    *,
    updated_at: str = "2026-05-28T00:00:00+00:00",
    pii_reviewed_at=None,
    pii_masked_at=None,
    quality_reviewed_by=None,
    storage_path: str | None = None,
):
    return {
        "id": rid,
        "sequence_order": seq,
        "pii_reviewed_at": pii_reviewed_at,
        "pii_masked_at": pii_masked_at,
        "quality_reviewed_by": quality_reviewed_by,
        "updated_at": updated_at,
        "storage_path": storage_path or f"utterances/sess1/{rid}.wav",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. dry_run=True → DELETE 0
# ─────────────────────────────────────────────────────────────────────────────


def test_dry_run_no_delete_call(fake_supabase, caplog):
    fake_supabase["utterances"] = {
        "select": [_row("u11", 11), _row("u12", 12)],
        "delete": [],
    }
    # H-loop labels probe → 비어있음
    fake_supabase["utterance_human_labels"] = {"select": [], "delete": []}

    with caplog.at_level(logging.INFO, logger="gpu_worker"):
        asyncio.run(_cleanup_stale_orphans(
            session_id="sess1",
            new_count=10,
            run_start_iso="2026-05-29T00:00:00+00:00",
            dry_run=True,
            min_ratio=0.5,
        ))

    # delete 호출 흔적 0
    assert "_last_delete" not in fake_supabase or "utterances" not in fake_supabase.get("_last_delete", {}), \
        "dry_run=True 인데 utterances.delete() 가 호출됨"
    # 로그에 dry_run=True 와 safe=2 표시
    msgs = [r.getMessage() for r in caplog.records]
    assert any("dry_run=True" in m for m in msgs), f"dry_run 로그 누락: {msgs}"
    assert any("safe=2" in m for m in msgs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. actual (dry_run=False) → safe 행만 DELETE
# ─────────────────────────────────────────────────────────────────────────────


def test_actual_deletes_safe_rows_only(fake_supabase):
    fake_supabase["utterances"] = {
        "select": [_row("u11", 11), _row("u12", 12), _row("u13", 13)],
        "delete": [{"id": "u11"}, {"id": "u12"}, {"id": "u13"}],
    }
    fake_supabase["utterance_human_labels"] = {"select": [], "delete": []}

    asyncio.run(_cleanup_stale_orphans(
        session_id="sess1",
        new_count=10,
        run_start_iso="2026-05-29T00:00:00+00:00",
        dry_run=False,
        min_ratio=0.5,
    ))

    last_delete_calls = fake_supabase.get("_last_delete", {}).get("utterances", [])
    assert last_delete_calls, "actual 모드인데 utterances.delete() 호출 흔적 없음"
    # in_("id", [...]) 인자 안에 u11/u12/u13 포함
    in_calls = [c for c in last_delete_calls if c[0] == "in_"]
    assert in_calls, "in_ 호출 누락"
    ids = in_calls[0][1][0][1]  # ((args, kwargs)) 구조 → args[1] = ids list
    assert set(ids) == {"u11", "u12", "u13"}, f"DELETE 대상 mismatch: {ids}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. pii_reviewed_at 보존
# ─────────────────────────────────────────────────────────────────────────────


def test_pii_reviewed_preserved(fake_supabase):
    fake_supabase["utterances"] = {
        "select": [
            _row("u11", 11, pii_reviewed_at="2026-05-25T00:00:00+00:00"),  # 보존
            _row("u12", 12),                                                # safe
        ],
        "delete": [{"id": "u12"}],
    }
    fake_supabase["utterance_human_labels"] = {"select": [], "delete": []}

    asyncio.run(_cleanup_stale_orphans(
        session_id="sess1", new_count=10,
        run_start_iso="2026-05-29T00:00:00+00:00",
        dry_run=False, min_ratio=0.5,
    ))

    in_calls = [c for c in fake_supabase.get("_last_delete", {}).get("utterances", []) if c[0] == "in_"]
    ids = in_calls[0][1][0][1]
    assert ids == ["u12"], f"pii_reviewed 행이 삭제됨: {ids}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. pii_masked_at 보존
# ─────────────────────────────────────────────────────────────────────────────


def test_pii_masked_preserved(fake_supabase):
    fake_supabase["utterances"] = {
        "select": [
            _row("u11", 11, pii_masked_at="2026-05-25T00:00:00+00:00"),
            _row("u12", 12),
        ],
        "delete": [{"id": "u12"}],
    }
    fake_supabase["utterance_human_labels"] = {"select": [], "delete": []}

    asyncio.run(_cleanup_stale_orphans(
        session_id="sess1", new_count=10,
        run_start_iso="2026-05-29T00:00:00+00:00",
        dry_run=False, min_ratio=0.5,
    ))
    ids = [c for c in fake_supabase.get("_last_delete", {}).get("utterances", []) if c[0] == "in_"][0][1][0][1]
    assert ids == ["u12"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. quality_reviewed_by 보존
# ─────────────────────────────────────────────────────────────────────────────


def test_quality_reviewed_preserved(fake_supabase):
    fake_supabase["utterances"] = {
        "select": [
            _row("u11", 11, quality_reviewed_by="admin-uuid"),
            _row("u12", 12),
        ],
        "delete": [{"id": "u12"}],
    }
    fake_supabase["utterance_human_labels"] = {"select": [], "delete": []}

    asyncio.run(_cleanup_stale_orphans(
        session_id="sess1", new_count=10,
        run_start_iso="2026-05-29T00:00:00+00:00",
        dry_run=False, min_ratio=0.5,
    ))
    ids = [c for c in fake_supabase.get("_last_delete", {}).get("utterances", []) if c[0] == "in_"][0][1][0][1]
    assert ids == ["u12"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. utterance_human_labels FK 라벨 존재 → 보존
# ─────────────────────────────────────────────────────────────────────────────


def test_human_labeled_preserved(fake_supabase):
    fake_supabase["utterances"] = {
        "select": [
            _row("u11", 11),
            _row("u12", 12),
            _row("u13", 13),
        ],
        "delete": [{"id": "u12"}, {"id": "u13"}],
    }
    # u11 에 라벨 존재 → 보존되어야 함
    fake_supabase["utterance_human_labels"] = {
        "select": [{"utterance_id": "u11"}],
        "delete": [],
    }

    asyncio.run(_cleanup_stale_orphans(
        session_id="sess1", new_count=10,
        run_start_iso="2026-05-29T00:00:00+00:00",
        dry_run=False, min_ratio=0.5,
    ))
    ids = [c for c in fake_supabase.get("_last_delete", {}).get("utterances", []) if c[0] == "in_"][0][1][0][1]
    assert set(ids) == {"u12", "u13"}, f"라벨 있는 u11 이 삭제 대상에 포함: {ids}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. new_count=0 → skip(silent-empty-done 가드)
# ─────────────────────────────────────────────────────────────────────────────


def test_new_count_zero_skipped(fake_supabase, caplog):
    fake_supabase["utterances"] = {"select": [_row("u11", 11)], "delete": []}

    with caplog.at_level(logging.INFO, logger="gpu_worker"):
        asyncio.run(_cleanup_stale_orphans(
            session_id="sess1", new_count=0,
            run_start_iso="2026-05-29T00:00:00+00:00",
            dry_run=False, min_ratio=0.5,
        ))

    # 어떤 query 도 실행 안 됨
    assert "_last_select" not in fake_supabase, "new_count=0 인데 SELECT 발생"
    assert "_last_delete" not in fake_supabase, "new_count=0 인데 DELETE 발생"
    assert any("silent-empty-done" in r.getMessage() for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ratio < min_ratio → skip + warning
# ─────────────────────────────────────────────────────────────────────────────


def test_ratio_guard_skips(fake_supabase, caplog):
    # new=2 / (2+10) = 0.166 < 0.5 → skip
    fake_supabase["utterances"] = {
        "select": [_row(f"u{i}", i) for i in range(3, 13)],
        "delete": [],
    }

    with caplog.at_level(logging.WARNING, logger="gpu_worker"):
        asyncio.run(_cleanup_stale_orphans(
            session_id="sess1", new_count=2,
            run_start_iso="2026-05-29T00:00:00+00:00",
            dry_run=False, min_ratio=0.5,
        ))

    assert "_last_delete" not in fake_supabase, "ratio guard 무시: DELETE 실행됨"
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("ratio" in m and "수동 검토" in m for m in msgs), f"ratio 가드 warning 누락: {msgs}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. cleanup 실패 → 후속 단계로 전파 안 됨(fail-closed)
# ─────────────────────────────────────────────────────────────────────────────


def test_cleanup_failure_does_not_propagate(caplog):
    # _run / _supabase 자체를 실패로 모킹
    async def _bad_run(*args, **kwargs):
        raise RuntimeError("simulated DB outage")

    with patch.object(worker_module, "_run", _bad_run), \
         caplog.at_level(logging.WARNING, logger="gpu_worker"):
        # 예외가 밖으로 새지 않아야 함
        asyncio.run(_cleanup_stale_orphans(
            session_id="sess1", new_count=10,
            run_start_iso="2026-05-29T00:00:00+00:00",
            dry_run=False, min_ratio=0.5,
        ))

    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("orphan cleanup failed" in m for m in msgs), \
        f"실패 warning 누락(또는 예외 전파됨): {msgs}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. updated_at >= run_start_iso → 본 turn 신규 갱신, 보존
# ─────────────────────────────────────────────────────────────────────────────


def test_updated_at_within_run_preserves(fake_supabase):
    # u11: run_start 직후 갱신 → 보존
    # u12: run_start 전 → safe
    fake_supabase["utterances"] = {
        "select": [
            _row("u11", 11, updated_at="2026-05-29T00:00:30+00:00"),  # >= run_start
            _row("u12", 12, updated_at="2026-05-28T00:00:00+00:00"),  # <  run_start
        ],
        "delete": [{"id": "u12"}],
    }
    fake_supabase["utterance_human_labels"] = {"select": [], "delete": []}

    asyncio.run(_cleanup_stale_orphans(
        session_id="sess1", new_count=10,
        run_start_iso="2026-05-29T00:00:00+00:00",
        dry_run=False, min_ratio=0.5,
    ))
    ids = [c for c in fake_supabase.get("_last_delete", {}).get("utterances", []) if c[0] == "in_"][0][1][0][1]
    assert ids == ["u12"], f"본 turn 신규 갱신 행이 삭제됨: {ids}"


# ─────────────────────────────────────────────────────────────────────────────
# 11. utterance_human_labels probe 실패 → 보수적 skip
# ─────────────────────────────────────────────────────────────────────────────


def test_labels_probe_failure_skips(caplog):
    call_count = {"n": 0}

    async def _flaky_run(fn, *args, **kwargs):
        # 1번 째 (utterances select) 정상, 2번 째 (labels probe) 실패
        call_count["n"] += 1
        if call_count["n"] == 1:
            class _R: data = [{"id": "u11", "sequence_order": 11,
                               "pii_reviewed_at": None, "pii_masked_at": None,
                               "quality_reviewed_by": None,
                               "updated_at": "2026-05-28T00:00:00+00:00",
                               "storage_path": "utterances/sess1/u11.wav"}]
            return _R()
        raise RuntimeError("utterance_human_labels not reachable")

    with patch.object(worker_module, "_run", _flaky_run), \
         caplog.at_level(logging.WARNING, logger="gpu_worker"):
        asyncio.run(_cleanup_stale_orphans(
            session_id="sess1", new_count=10,
            run_start_iso="2026-05-29T00:00:00+00:00",
            dry_run=False, min_ratio=0.5,
        ))

    # labels probe 실패 → cleanup 자체가 보수적 skip(DELETE 0)
    assert call_count["n"] == 2, f"labels probe 가 호출되지 않음(또는 추가 호출): n={call_count['n']}"
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("utterance_human_labels probe" in m and "보수적" in m for m in msgs), \
        f"보수적 skip warning 누락: {msgs}"


# ─────────────────────────────────────────────────────────────────────────────
# 12. 후보 0 → noop (정상)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_candidates_noop(fake_supabase):
    fake_supabase["utterances"] = {"select": [], "delete": []}

    asyncio.run(_cleanup_stale_orphans(
        session_id="sess1", new_count=10,
        run_start_iso="2026-05-29T00:00:00+00:00",
        dry_run=False, min_ratio=0.5,
    ))
    # SELECT 만 발생, DELETE 없음.
    assert fake_supabase.get("_last_select", {}).get("utterances"), "후보 select 가 발생하지 않음"
    assert "_last_delete" not in fake_supabase, "후보 0 인데 DELETE 발생"


# ─────────────────────────────────────────────────────────────────────────────
# 13. env unset → cleanup 호출 진입 안 함 (모듈 상수 검증)
# ─────────────────────────────────────────────────────────────────────────────


def test_module_constants_default_off_dry():
    """모듈 로드 시 기본 env 로 worker 가 import 됐을 때 ENABLED=False / DRY_RUN=True."""
    # conftest 이전에 env unset 으로 import 했으므로 기본값 확인
    assert worker_module.ORPHAN_CLEANUP_ENABLED is False, \
        "기본 ORPHAN_CLEANUP_ENABLED 가 False 아님 — 머지 직후 운영 영향 위험"
    assert worker_module.ORPHAN_CLEANUP_DRY_RUN is True, \
        "기본 ORPHAN_CLEANUP_DRY_RUN 가 True 아님 — 머지 직후 실 DELETE 위험"
    assert worker_module.ORPHAN_CLEANUP_MIN_RATIO == 0.5
