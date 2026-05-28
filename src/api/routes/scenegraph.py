"""Scene graph API route."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.task_queue import TaskQueue
from src.services import ModelServices

router = APIRouter(tags=["scenegraph"])

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}


def _svc(r: Request) -> ModelServices:
    return r.app.state.services


def _tq(r: Request) -> TaskQueue:
    return r.app.state.task_queue


@router.post("/scenegraph")
async def start_scenegraph(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    output_type: str = Form("json"),
    temperature: Optional[float] = Form(None),
    num_frames: Optional[int] = Form(None),
    mode: str = Form("high"),
    prompt_override: Optional[str] = Form(None),
):
    """Submit a scene-graph task; returns {task_id} for polling."""
    services = _svc(request)
    task_queue = _tq(request)
    cfg = services.cfg

    if not (file and file.filename) and not text:
        return JSONResponse({"error": "Provide a file or text."}, status_code=400)

    tmp_dir = cfg["paths"]["tmp_dir"]
    out_dir = cfg["paths"]["output_dir"]

    # Save upload
    media_path: Optional[str] = None
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1] or ".bin"
        media_path = tempfile.mktemp(suffix=ext, dir=tmp_dir)
        async with aiofiles.open(media_path, "wb") as fh:
            while chunk := await file.read(1 << 20):
                await fh.write(chunk)

    # Pre-allocate output path for overlay mode
    out_path: Optional[str] = None
    if output_type == "overlay" and media_path:
        ext2 = os.path.splitext(media_path)[1].lower()
        out_ext = ".png" if ext2 in _IMAGE_EXTS else ".mp4"
        out_path = tempfile.mktemp(suffix=f"_overlay{out_ext}", dir=out_dir)

    sg = services.sg_pipeline
    _text = text or ""
    _otype = output_type
    _mode = mode if mode in ("high", "low") else "high"
    _prompt_override = prompt_override or None

    def _run():
        try:
            raw = sg.process(
                media_path=media_path,
                text=_text,
                output_type=_otype,
                output_path=out_path,
                temperature=temperature,
                num_frames=num_frames,
                mode=_mode,
                prompt_override=_prompt_override,
            )
            return {
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "triplets": [
                            {"subject": t[0], "relation": t[1], "object": t[2]}
                            for t in seg.get("triplets", [])
                        ],
                    }
                    for seg in raw.get("segments", [])
                ],
                "overlay_path": raw.get("overlay_path"),
                "overlay_error": raw.get("overlay_error"),
            }
        finally:
            if media_path and os.path.isfile(media_path):
                try:
                    os.remove(media_path)
                except OSError:
                    pass

    task_id = await task_queue.submit(_run)
    return {"task_id": task_id}
