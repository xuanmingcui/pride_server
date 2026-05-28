"""Generic task status and file-download endpoints."""
from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from src.api.task_queue import TaskQueue

router = APIRouter(tags=["tasks"])


def _tq(request: Request) -> TaskQueue:
    return request.app.state.task_queue


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    """Poll a task for its status and result."""
    task = _tq(request).get(task_id)
    if task is None:
        return JSONResponse({"error": "Task not found."}, status_code=404)
    return task.to_dict()


@router.get("/tasks/{task_id}/file")
async def download_task_file(task_id: str, request: Request):
    """Stream the overlay file (video/image) produced by a completed task."""
    task = _tq(request).get(task_id)
    if task is None:
        return JSONResponse({"error": "Task not found."}, status_code=404)
    if task.status != "done":
        return JSONResponse({"error": "Task not completed yet."}, status_code=400)
    if not isinstance(task.result, dict):
        return JSONResponse({"error": "No file for this task."}, status_code=404)
    fpath = task.result.get("overlay_path")
    if not fpath or not os.path.isfile(fpath):
        return JSONResponse({"error": "File not found on server."}, status_code=404)
    return FileResponse(fpath, filename=os.path.basename(fpath))
