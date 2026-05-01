import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from app import config
from app.models.schemas import TaskInfo, TaskStatus

logger = logging.getLogger(__name__)

MAX_COMPLETED_AGE_SEC = 3600  # 완료 작업 1시간 후 자동 삭제
MAX_STORE_SIZE = 100          # 최대 보관 건수


class JobStore:
    """Thread-safe in-memory job state store with TTL cleanup."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._ensure_results_dir()

    def _ensure_results_dir(self) -> Path:
        """Ensure results directory exists. Use /dev/shm if available, fallback to /tmp."""
        target_dir = config.RESULTS_DIR
        if target_dir.parent.name == "shm":
            # /dev/shm case: check if /dev/shm exists
            shm_parent = Path("/dev/shm")
            if not shm_parent.exists():
                logger.warning("/dev/shm not available, falling back to /tmp")
                target_dir = Path("/tmp/stt-results")
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def _get_results_dir(self) -> Path:
        """Get the actual results directory (may be different from config if /dev/shm unavailable)."""
        if Path("/dev/shm").exists():
            return config.RESULTS_DIR
        return Path("/tmp/stt-results")

    def create(self, task_id: str) -> TaskInfo:
        now = time.time()
        task = TaskInfo(task_id=task_id, status=TaskStatus.pending, queued_at=now)
        with self._lock:
            self._tasks[task_id] = task
            self._timestamps[task_id] = now
            self._cleanup_expired()
        return task

    def get(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def active_count(self) -> int:
        """pending + processing 상태인 작업 수 (큐 백프레셔용)."""
        with self._lock:
            return sum(
                1 for t in self._tasks.values()
                if t.status in (TaskStatus.pending, TaskStatus.processing)
            )

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                # Terminal state guard: 이미 completed/failed면 status 변경 금지 (race overwrite 방지)
                if task.status in (TaskStatus.completed, TaskStatus.failed):
                    logger.warning(
                        "[%s] update_status race: terminal=%s → ignored new=%s",
                        task_id, task.status.value, status.value,
                    )
                    return
                self._tasks[task_id] = task.model_copy(update={"status": status})

    def set_result(self, task_id: str, result: dict) -> None:
        with self._lock:
            if task_id not in self._tasks:
                # Race: task evicted before result arrived. 호출자가 검증/재시도 결정.
                logger.error(
                    "[%s] set_result race: task evicted before completion — result lost",
                    task_id,
                )
                return
            task = self._tasks[task_id]
            # Terminal state guard: 이미 completed/failed인 task에 set_result 시도 차단
            if task.status in (TaskStatus.completed, TaskStatus.failed):
                logger.error(
                    "[%s] set_result race: terminal=%s → duplicate set_result ignored",
                    task_id, task.status.value,
                )
                return
            self._tasks[task_id] = task.model_copy(
                update={"status": TaskStatus.completed, "result": result}
            )

    def set_error(self, task_id: str, error: str) -> None:
        with self._lock:
            if task_id not in self._tasks:
                logger.error(
                    "[%s] set_error race: task evicted before error reported — error lost",
                    task_id,
                )
                return
            task = self._tasks[task_id]
            # Terminal state guard
            if task.status in (TaskStatus.completed, TaskStatus.failed):
                logger.warning(
                    "[%s] set_error race: terminal=%s → duplicate set_error ignored",
                    task_id, task.status.value,
                )
                return
            self._tasks[task_id] = task.model_copy(
                update={"status": TaskStatus.failed, "error": error}
            )

    def update_gpu_acquired(self, task_id: str) -> None:
        """GPU 세마포어 획득 시각 기록 (관측용)."""
        now = time.time()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                self._tasks[task_id] = task.model_copy(update={"gpu_acquired_at": now})

    def update_gpu_released(self, task_id: str) -> None:
        """GPU 세마포어 해제 시각 기록 (관측용)."""
        now = time.time()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                self._tasks[task_id] = task.model_copy(update={"gpu_released_at": now})

    def position_of(self, task_id: str) -> Optional[int]:
        """queued_at 기준 1-based 대기 순번을 반환한다.

        대기 집합 = pending + processing 상태 (GPU 점유 중 포함).
        complete/failed 이거나 queued_at 없으면 None.
        """
        with self._lock:
            target = self._tasks.get(task_id)
            if target is None or target.queued_at is None:
                return None
            if target.status not in (TaskStatus.pending, TaskStatus.processing):
                return None
            waiting = [
                t for t in self._tasks.values()
                if t.status in (TaskStatus.pending, TaskStatus.processing)
                and t.queued_at is not None
            ]
            waiting.sort(key=lambda t: t.queued_at)  # type: ignore[return-value,arg-type]
            for idx, t in enumerate(waiting):
                if t.task_id == task_id:
                    return idx + 1
            return None

    def queue_snapshot(self) -> dict:
        """GPU 세마포어 점유 상황 + 대기 큐 상태 스냅샷 (관측용).

        Returns:
            dict with keys: gpu_busy, current_task_id, queue_depth, waiting_task_ids.
        """
        with self._lock:
            current: Optional[TaskInfo] = None
            pending: list[TaskInfo] = []
            for t in self._tasks.values():
                if t.status not in (TaskStatus.pending, TaskStatus.processing):
                    continue
                # GPU 점유 중: acquired 있고 released 없음
                if t.gpu_acquired_at is not None and t.gpu_released_at is None:
                    # 세마포어(1) 가정 하에 현재 점유 task는 최대 1개
                    if current is None or (
                        t.gpu_acquired_at
                        and current.gpu_acquired_at
                        and t.gpu_acquired_at > current.gpu_acquired_at
                    ):
                        current = t
                else:
                    pending.append(t)
            pending.sort(key=lambda x: x.queued_at or 0.0)
            return {
                "gpu_busy": current is not None,
                "current_task_id": current.task_id if current else None,
                "queue_depth": len(pending),
                "waiting_task_ids": [t.task_id for t in pending],
            }


    def set_audio(self, task_id: str, audio_files: dict[str, bytes]) -> None:
        """Write audio files to disk under per-task subdirectory."""
        results_dir = self._get_results_dir()
        task_audio_dir = results_dir / task_id
        task_audio_dir.mkdir(parents=True, exist_ok=True)

        for filename, wav_bytes in audio_files.items():
            audio_path = task_audio_dir / filename
            audio_path.write_bytes(wav_bytes)
            logger.info("[%s] 오디오 저장: %s (%d bytes)", task_id, filename, len(wav_bytes))

    def get_audio(self, task_id: str, filename: str) -> Path | None:
        """Return Path to audio file if it exists, None otherwise."""
        results_dir = self._get_results_dir()
        task_dir = results_dir / task_id
        audio_path = (task_dir / filename).resolve()
        # Path traversal 방지: 결과 경로가 task 디렉토리 내부인지 확인
        if not str(audio_path).startswith(str(task_dir.resolve())):
            logger.warning("[%s] Path traversal 시도 차단: %s", task_id, filename)
            return None
        if audio_path.exists() and audio_path.is_file():
            return audio_path
        logger.warning("[%s] Audio not found: %s", task_id, filename)
        return None


    def _cleanup_expired(self) -> None:
        """Remove completed/failed tasks older than MAX_COMPLETED_AGE_SEC. Must hold lock.

        Also marks pending/processing tasks as failed if they exceed MAX_PROCESSING_AGE_SEC
        to prevent stuck jobs from blocking the queue indefinitely.
        """
        now = time.time()
        expired = []
        for task_id, ts in self._timestamps.items():
            task = self._tasks.get(task_id)
            if task is None:
                expired.append(task_id)
                continue
            age = now - ts
            if task.status in (TaskStatus.pending, TaskStatus.processing):
                if age > config.MAX_PROCESSING_AGE_SEC:
                    logger.warning(
                        "[%s] Stuck job detected (status=%s, age=%.0fs) — marking failed",
                        task_id, task.status.value, age,
                    )
                    self._tasks[task_id] = task.model_copy(
                        update={"status": TaskStatus.failed, "error": "processing_timeout"}
                    )
            if age > MAX_COMPLETED_AGE_SEC and task.status in (TaskStatus.completed, TaskStatus.failed):
                expired.append(task_id)

        # Over-size eviction: pending/processing은 절대 evict 안 함 (race outlier 방지).
        # 완료/실패 task만 가장 오래된 것부터 제거. MAX_STORE_SIZE는 soft-cap.
        if len(self._tasks) > MAX_STORE_SIZE:
            terminal_by_age = sorted(
                (
                    (tid, ts) for tid, ts in self._timestamps.items()
                    if (t := self._tasks.get(tid)) is not None
                    and t.status in (TaskStatus.completed, TaskStatus.failed)
                ),
                key=lambda x: x[1],
            )
            need_evict = len(self._tasks) - MAX_STORE_SIZE
            for task_id, _ in terminal_by_age[:need_evict]:
                if task_id not in expired:
                    expired.append(task_id)
            # active(pending/processing) > MAX_STORE_SIZE면 soft-cap 초과 허용 (백프레셔는 active_count로)
            active_only = len(self._tasks) - len(expired)
            if active_only > MAX_STORE_SIZE:
                logger.warning(
                    "JobStore over soft-cap: active=%d > MAX_STORE_SIZE=%d (백프레셔 active_count로 통제)",
                    active_only, MAX_STORE_SIZE,
                )

        results_dir = self._get_results_dir()
        for task_id in expired:
            self._tasks.pop(task_id, None)
            self._timestamps.pop(task_id, None)
            # Remove per-task audio directory
            task_audio_dir = results_dir / task_id
            if task_audio_dir.exists():
                shutil.rmtree(task_audio_dir, ignore_errors=True)
                logger.info("[%s] 오디오 디렉토리 삭제", task_id)

        if expired:
            logger.info("JobStore cleanup: %d건 제거 (남은: %d건)", len(expired), len(self._tasks))


job_store = JobStore()
