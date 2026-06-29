"""Background task queue for long-running ML operations.

Tasks are submitted as zero-argument callables (use closures to capture args).
The single worker mirrors the single-thread executor used for ML inference.
Results are retained for TTL_SECONDS after completion, then cleaned up.
"""
from __future__ import annotations

import asyncio
import inspect
import functools
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional

log = logging.getLogger("pride.tasks")

TTL_SECONDS = 3600  # keep completed task results for 1 hour


@dataclass
class TaskInfo:
    task_id: str
    status: Literal["pending", "running", "done", "error"] = "pending"
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    # {percent: 0-100 float, stage: short label, detail: optional extra text}
    progress: Dict[str, Any] = field(
        default_factory=lambda: {"percent": 0.0, "stage": "queued", "detail": "", "stages": []}
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result if self.status == "done" else None,
            "error": self.error,
            "progress": self.progress,
        }


def _accepts_report(fn: Callable) -> bool:
    """True if ``fn`` takes at least one positional parameter (the reporter)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in params
    )


class TaskQueue:
    """Single-worker async task queue backed by a ThreadPoolExecutor."""

    def __init__(self, executor: ThreadPoolExecutor):
        self._executor = executor
        self._tasks: Dict[str, TaskInfo] = {}
        self._queue: Optional[asyncio.Queue] = None

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        asyncio.create_task(self._worker())
        asyncio.create_task(self._cleanup_loop())

    async def submit(self, fn: Callable[[], Any]) -> str:
        """Submit a zero-argument callable; returns task_id immediately."""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = TaskInfo(task_id=task_id)
        await self._queue.put((task_id, fn))
        return task_id

    def get(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    async def _worker(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            task_id, fn = await self._queue.get()
            info = self._tasks[task_id]
            info.status = "running"
            info.progress = {"percent": 1.0, "stage": "starting", "detail": ""}

            def report(percent: Optional[float] = None,
                       stage: Optional[str] = None,
                       detail: Optional[str] = None,
                       stages: Optional[Any] = None) -> None:
                # Called from the worker thread; plain dict writes are fine here.
                if percent is not None:
                    info.progress["percent"] = max(0.0, min(100.0, float(percent)))
                if stage is not None:
                    info.progress["stage"] = stage
                if detail is not None:
                    info.progress["detail"] = detail
                if stages is not None:
                    info.progress["stages"] = stages

            call = functools.partial(fn, report) if _accepts_report(fn) else fn
            try:
                info.result = await loop.run_in_executor(self._executor, call)
                info.status = "done"
                # Preserve the stage list (mark every stage complete) so the UI
                # shows the finished checklist rather than a blank bar.
                done_stages = info.progress.get("stages") or []
                for s in done_stages:
                    s["state"] = "done"
                    s["percent"] = 100.0
                info.progress = {"percent": 100.0, "stage": "done",
                                 "detail": "", "stages": done_stages}
                log.info("Task %s done.", task_id)
            except Exception as exc:
                import traceback
                info.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                info.status = "error"
                log.error("Task %s failed: %s", task_id, exc)
            finally:
                self._queue.task_done()

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(600)
            cutoff = time.time() - TTL_SECONDS
            to_delete = [
                tid for tid, t in self._tasks.items()
                if t.status in ("done", "error") and t.created_at < cutoff
            ]
            for tid in to_delete:
                task = self._tasks.pop(tid, None)
                if task and isinstance(task.result, dict):
                    fpath = task.result.get("overlay_path")
                    if fpath and os.path.isfile(fpath):
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
            if to_delete:
                log.info("Cleaned up %d expired task(s).", len(to_delete))
