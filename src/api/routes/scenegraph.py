"""Scene graph API route."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import aiofiles
from fastapi import APIRouter, Body, File, Form, Request, UploadFile
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
    fps: Optional[float] = Form(None),
    mode: str = Form("high"),
    system_prompt_override: Optional[str] = Form(None),
    user_prompt_override: Optional[str] = Form(None),
    raw_output: bool = Form(False),
    normalize: Optional[bool] = Form(None),
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
    _prompt_override = (
        {"system": system_prompt_override, "user": user_prompt_override}
        if (system_prompt_override or user_prompt_override) else None
    )

    def _run():
        try:
            raw = sg.process(
                media_path=media_path,
                text=_text,
                output_type=_otype,
                output_path=out_path,
                temperature=temperature,
                fps=fps,
                mode=_mode,
                prompt_override=_prompt_override,
                raw_output=raw_output,
                normalize=normalize,
            )
            raw_segs = raw.get("segments", [])
            is_temporal = bool(raw_segs)

            def _fmt_item(t):
                # Quintuples (s, r, o, start_sec, end_sec) come from video paths;
                # triplets (s, r, o) come from image / text-only paths.
                if len(t) >= 5:
                    return {
                        "subject":   t[0],
                        "relation":  t[1],
                        "object":    t[2],
                        "start_sec": float(t[3]),
                        "end_sec":   float(t[4]),
                    }
                return {"subject": t[0], "relation": t[1], "object": t[2]}

            def _fmt_seg(seg):
                out = {
                    "start":    seg.get("start"),
                    "end":      seg.get("end"),
                    "triplets": [_fmt_item(t) for t in seg.get("triplets", [])],
                }
                if "raw_text" in seg:
                    out["raw_text"] = seg["raw_text"]
                return out

            if is_temporal:
                segments_out = [_fmt_seg(seg) for seg in raw_segs]
            else:
                flat = raw.get("triplets", [])
                base = {
                    "start": None, "end": None,
                    "triplets": [_fmt_item(t) for t in flat],
                }
                if "raw_text" in raw:
                    base["raw_text"] = raw["raw_text"]
                segments_out = [base] if (flat or "raw_text" in raw) else []

            return {
                "is_temporal":  is_temporal,
                "raw_output":   raw_output,
                "segments":     segments_out,
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


@router.post("/scenegraph/normalize")
async def normalize_scenegraph(
    request: Request,
    payload: dict = Body(...),
):
    """Run the refinement pass on already-generated scene graph segments.

    Beyond entity normalization, the refinement pass also tightens vague
    relations, merges near-duplicates (taking the union of their time
    windows for quintuples), and drops low-quality / trivially-implied rows.

    Body: ``{"segments": [{start, end, triplets: [{subject, relation, object[, start_sec, end_sec]}]}]}``.
    Returns ``{task_id}``; poll ``GET /api/tasks/{id}`` for the refined
    segments in the same shape.
    """
    services = _svc(request)
    task_queue = _tq(request)

    segments = payload.get("segments")
    if not isinstance(segments, list):
        return JSONResponse({"error": "Body must include 'segments' (list)."}, status_code=400)

    sg = services.sg_pipeline

    def _run():
        normalized = sg.normalize_segments(segments)
        return {"segments": normalized}

    task_id = await task_queue.submit(_run)
    return {"task_id": task_id}
