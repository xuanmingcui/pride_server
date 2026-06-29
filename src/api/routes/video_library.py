"""Video Library API routes.

Build and query a structured index of scene-graph rows keyed by source video.
Indexing (embedding N rows, optionally generating the scene graph first) goes
through the task queue; listing / search / delete are fast and run inline.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import APIRouter, Body, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.api.task_queue import TaskQueue
from src.services import ModelServices

router = APIRouter(prefix="/videos", tags=["videos"])


def _svc(r: Request) -> ModelServices:
    return r.app.state.services


def _tq(r: Request) -> TaskQueue:
    return r.app.state.task_queue


async def _run(svc: ModelServices, fn):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(svc.executor, fn)


def _flatten_sg(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a scene-graph ``process()`` result into uniform row dicts.

    Handles both temporal video output (segments of 5-tuple quintuples) and
    flat image/text output (3-tuple triplets).
    """
    rows: List[Dict[str, Any]] = []

    def _emit(t):
        # Two shapes flow through here: tuples/lists from sg_pipeline.process()
        # ((s, r, o[, start, end])) and dicts from the frontend "Add to Library"
        # path ({subject, relation, object[, start_sec, end_sec]}).
        if isinstance(t, dict):
            row = {"subject": t.get("subject"), "relation": t.get("relation"),
                   "object": t.get("object")}
            if t.get("start_sec") is not None:
                row["start_sec"] = float(t["start_sec"])
            if t.get("end_sec") is not None:
                row["end_sec"] = float(t["end_sec"])
            rows.append(row)
        elif isinstance(t, (list, tuple)) and len(t) >= 5:
            rows.append({"subject": t[0], "relation": t[1], "object": t[2],
                         "start_sec": float(t[3]), "end_sec": float(t[4])})
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            rows.append({"subject": t[0], "relation": t[1], "object": t[2]})

    segs = result.get("segments") or []
    if segs:
        for seg in segs:
            for t in seg.get("triplets", []):
                _emit(t)
    else:
        for t in result.get("triplets", []):
            _emit(t)
    return rows


# ── Listing ───────────────────────────────────────────────────────────────────

@router.get("")
async def list_videos(request: Request):
    svc = _svc(request)
    videos = await _run(svc, svc.video_index.list_videos)
    return {"videos": videos, "total": len(videos)}


# ── Search ────────────────────────────────────────────────────────────────────

@router.post("/search")
async def search_videos(request: Request, payload: dict = Body(...)):
    svc = _svc(request)
    query = (payload.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "Provide a query."}, status_code=400)
    top_k = int(payload.get("top_k") or 10)
    facet = (payload.get("facet") or "").strip() or None
    keyword = (payload.get("keyword") or "").strip() or None
    raw_min = payload.get("min_score")
    min_score = float(raw_min) if raw_min is not None and str(raw_min) != "" else None
    results = await _run(
        svc,
        lambda: svc.video_index.search(
            query, max_videos=top_k, facet=facet, keyword=keyword, min_score=min_score,
        ),
    )
    return {"query": query, "results": results}


# ── Indexing ──────────────────────────────────────────────────────────────────

@router.post("/index")
async def index_video(
    request: Request,
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    source: str = Form("user"),
    segments: Optional[str] = Form(None),   # JSON: pre-generated scene graph
    mode: str = Form("high"),
    fps: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None),
    normalize: Optional[bool] = Form(None),
):
    """Index a video into the structured library.

    Two flows share this endpoint:
      * Pre-generated graph ("Add to Library" from the Scene Graph tab): pass
        ``segments`` (JSON of ``{segments|triplets}``-style rows) plus the file.
        We skip generation and only embed/store.
      * Self-contained (Video Library tab): pass just the file; we run the
        scene-graph pipeline first, then index its output.

    Returns ``{task_id}``; poll ``GET /api/tasks/{id}``.
    """
    svc = _svc(request)
    task_queue = _tq(request)
    cfg = svc.cfg

    pre_rows: Optional[List[Dict[str, Any]]] = None
    if segments:
        try:
            payload = json.loads(segments)
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid 'segments' JSON."}, status_code=400)
        # Accept either a {segments:[...]} / {triplets:[...]} wrapper or a bare list.
        if isinstance(payload, dict):
            pre_rows = _flatten_sg(payload)
        elif isinstance(payload, list):
            pre_rows = _flatten_sg({"segments": payload}) or _flatten_sg({"triplets": payload})

    if not (file and file.filename) and pre_rows is None:
        return JSONResponse(
            {"error": "Provide a video file (and optionally a pre-generated scene graph)."},
            status_code=400,
        )

    tmp_dir = cfg["paths"]["tmp_dir"]
    media_path: Optional[str] = None
    ext: Optional[str] = None
    disp_title = title or (file.filename if file and file.filename else "video")
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        media_path = tempfile.mktemp(suffix=ext, dir=tmp_dir)
        async with aiofiles.open(media_path, "wb") as fh:
            while chunk := await file.read(1 << 20):
                await fh.write(chunk)

    _mode = mode if mode in ("high", "low") else "high"

    def _run_index():
        try:
            rows = pre_rows
            if rows is None:
                result = svc.sg_pipeline.process(
                    media_path=media_path,
                    text="",
                    output_type="json",
                    temperature=temperature,
                    fps=fps,
                    mode=_mode,
                    normalize=normalize,
                )
                rows = _flatten_sg(result)
            record = svc.video_index.index_video(
                rows=rows,
                title=disp_title,
                media_src_path=media_path,
                ext=ext,
                source=source,
            )
            return record
        finally:
            if media_path and os.path.isfile(media_path):
                try:
                    os.remove(media_path)
                except OSError:
                    pass

    task_id = await task_queue.submit(_run_index)
    return {"task_id": task_id}


# ── Media playback ────────────────────────────────────────────────────────────

@router.get("/{video_id}/media")
async def video_media(video_id: str, request: Request):
    svc = _svc(request)
    path = svc.video_index.media_path(video_id)
    if not path:
        return JSONResponse({"error": "No media for this video."}, status_code=404)
    # FileResponse honours HTTP Range requests, so the player can seek.
    return FileResponse(path, filename=os.path.basename(path))


# ── Deletion ──────────────────────────────────────────────────────────────────

@router.delete("/{video_id}")
async def delete_video(video_id: str, request: Request):
    svc = _svc(request)
    ok = await _run(svc, lambda: svc.video_index.delete_video(video_id))
    if not ok:
        return JSONResponse({"error": "Video not found."}, status_code=404)
    return {"deleted": video_id}
