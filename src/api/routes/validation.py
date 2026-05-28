"""Validation API route."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.task_queue import TaskQueue
from src.services import ModelServices

router = APIRouter(tags=["validation"])


def _svc(r: Request) -> ModelServices:
    return r.app.state.services


def _tq(r: Request) -> TaskQueue:
    return r.app.state.task_queue


@router.post("/validate")
async def start_validate(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    top_k: Optional[int] = Form(None),
    prompt_override: Optional[str] = Form(None),
):
    """Submit a validation task; returns {task_id} for polling."""
    services = _svc(request)
    task_queue = _tq(request)
    cfg = services.cfg

    if not (file and file.filename) and not text:
        return JSONResponse({"error": "Provide a file or text."}, status_code=400)

    db_name = database or cfg["validation"]["default_db"]
    tmp_dir = cfg["paths"]["tmp_dir"]

    media_path: Optional[str] = None
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1] or ".bin"
        media_path = tempfile.mktemp(suffix=ext, dir=tmp_dir)
        async with aiofiles.open(media_path, "wb") as fh:
            while chunk := await file.read(1 << 20):
                await fh.write(chunk)

    vp = services.val_pipeline
    _text = text or ""
    _prompt_override = prompt_override or None

    def _run():
        try:
            report = vp.validate(
                database=db_name,
                media_path=media_path,
                text=_text,
                top_k=top_k,
                prompt_override=_prompt_override,
            )
            return report.to_dict()
        finally:
            if media_path and os.path.isfile(media_path):
                try:
                    os.remove(media_path)
                except OSError:
                    pass

    task_id = await task_queue.submit(_run)
    return {"task_id": task_id}
