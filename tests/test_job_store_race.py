"""JobStore race condition 회귀 테스트 (Sprint 1 Day 1 — production task_id 매칭 fix).

배경:
- Sample(1) 100건 측정에서 14건 CER>100% outlier 발생
- 단독 호출 + 1.5s sleep 패턴으로 재처리 시 0건 — race condition 확실
- root cause: set_result terminal guard 부재 + cleanup_expired MAX_STORE_SIZE 초과 시
  pending/processing task evict
"""
import threading
import time

import pytest

from app.core.job_store import JobStore, MAX_STORE_SIZE
from app.models.schemas import TaskStatus


@pytest.fixture
def store():
    """매 테스트마다 깨끗한 JobStore 인스턴스."""
    return JobStore()


# ─────────────────────────────────────────────────────────────────────
# Fix 1 — set_result/set_error/update_status terminal-state guard
# ─────────────────────────────────────────────────────────────────────


def test_set_result_terminal_guard_ignores_duplicate(store):
    """set_result 두 번 호출 시 두 번째는 무시 (첫 결과 보존)."""
    store.create("t1")
    store.set_result("t1", {"text": "first"})
    store.set_result("t1", {"text": "second"})  # duplicate — should be ignored

    task = store.get("t1")
    assert task.status == TaskStatus.completed
    assert task.result == {"text": "first"}, "duplicate set_result가 첫 결과를 덮어쓰면 안 됨"


def test_set_error_after_completed_ignored(store):
    """set_result 후 set_error 호출 시 set_error 무시 (terminal guard)."""
    store.create("t1")
    store.set_result("t1", {"text": "ok"})
    store.set_error("t1", "race error")

    task = store.get("t1")
    assert task.status == TaskStatus.completed
    assert task.result == {"text": "ok"}
    assert task.error is None, "completed task에 set_error 시도가 status를 덮어쓰면 안 됨"


def test_set_result_after_failed_ignored(store):
    """set_error 후 set_result 호출 시 set_result 무시."""
    store.create("t1")
    store.set_error("t1", "boom")
    store.set_result("t1", {"text": "late"})

    task = store.get("t1")
    assert task.status == TaskStatus.failed
    assert task.error == "boom"
    assert task.result is None, "failed task에 set_result 시도가 result를 채우면 안 됨"


def test_update_status_after_completed_ignored(store):
    """update_status로 terminal task를 다시 processing으로 되돌릴 수 없음."""
    store.create("t1")
    store.set_result("t1", {"text": "ok"})
    store.update_status("t1", TaskStatus.processing)  # race rollback 시도

    task = store.get("t1")
    assert task.status == TaskStatus.completed, "terminal status 롤백 차단 필수"


def test_set_result_evicted_task_silent(store, caplog):
    """task가 evict된 후 set_result 호출 시 조용히 실패 + error 로깅."""
    # task 생성 후 강제 제거 (evict 시뮬레이션)
    store.create("t1")
    with store._lock:
        store._tasks.pop("t1", None)
        store._timestamps.pop("t1", None)

    # set_result는 race로 처리 — 예외 없이 로그만 남김
    store.set_result("t1", {"text": "lost"})
    assert store.get("t1") is None
    assert any("evicted" in r.message for r in caplog.records), "evict race는 로깅돼야 함"


# ─────────────────────────────────────────────────────────────────────
# cleanup_expired — pending/processing 보호
# ─────────────────────────────────────────────────────────────────────


def test_cleanup_does_not_evict_pending_under_pressure(store):
    """MAX_STORE_SIZE 초과 시에도 pending/processing task는 evict 금지 (real outlier 원인)."""
    # MAX_STORE_SIZE + 10개 pending task 강제 생성
    for i in range(MAX_STORE_SIZE + 10):
        store.create(f"pending_{i}")

    # 모든 pending이 살아있어야 함 — 이전 코드는 가장 오래된 10개를 evict했음
    surviving = sum(
        1 for tid in (f"pending_{i}" for i in range(MAX_STORE_SIZE + 10))
        if store.get(tid) is not None
    )
    assert surviving == MAX_STORE_SIZE + 10, (
        f"pending task가 evict됨: {surviving}/{MAX_STORE_SIZE + 10}"
    )


def test_cleanup_evicts_completed_oldest_first(store):
    """soft-cap 초과 시 completed task만 가장 오래된 것부터 evict."""
    # MAX_STORE_SIZE 만큼 completed 만들고 추가로 pending 1개
    for i in range(MAX_STORE_SIZE):
        store.create(f"done_{i}")
        store.set_result(f"done_{i}", {"idx": i})
    # _cleanup_expired 트리거 — pending 신규 추가 시
    store.create("new_pending")

    # 가장 오래된 done_0 evict
    assert store.get("done_0") is None, "가장 오래된 completed가 evict돼야 함"
    assert store.get("done_99") is not None, "최근 completed는 보존돼야 함"
    assert store.get("new_pending") is not None, "새 pending은 보존돼야 함"


# ─────────────────────────────────────────────────────────────────────
# Concurrent stress — 동시 set_result 매칭 정확성
# ─────────────────────────────────────────────────────────────────────


def test_concurrent_set_result_matching_correct(store):
    """동시 100 thread가 자기 task_id에만 set_result — 결과 섞임 없어야 함."""
    N = 100
    for i in range(N):
        store.create(f"task_{i}")

    barrier = threading.Barrier(N)

    def worker(idx: int):
        barrier.wait()  # 모든 thread 동시 진입
        store.set_result(f"task_{idx}", {"idx": idx, "marker": f"unique_{idx}"})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 모든 task가 자기 idx로 정확히 매칭됐어야 함
    for i in range(N):
        task = store.get(f"task_{i}")
        assert task is not None, f"task_{i} 사라짐"
        assert task.status == TaskStatus.completed
        assert task.result["idx"] == i, f"task_{i} result 섞임: {task.result}"
        assert task.result["marker"] == f"unique_{i}"


def test_concurrent_duplicate_set_result_first_wins(store):
    """같은 task_id에 두 thread가 동시에 set_result — 한 개만 채택, 다른 하나는 silent ignore."""
    store.create("contested")
    barrier = threading.Barrier(2)
    results_log = []

    def worker(value: str):
        barrier.wait()
        store.set_result("contested", {"value": value})
        results_log.append(value)

    t1 = threading.Thread(target=worker, args=("first",))
    t2 = threading.Thread(target=worker, args=("second",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    final = store.get("contested")
    assert final.status == TaskStatus.completed
    # 둘 중 하나만 채택 — 나머지는 terminal guard로 무시
    assert final.result["value"] in ("first", "second")
