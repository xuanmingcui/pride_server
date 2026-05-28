"""Database management API routes.

Lightweight CRUD operations run directly in the ML executor (fast).
add_facts goes through the task queue because embedding N texts can take seconds.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Body, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.task_queue import TaskQueue
from src.services import ModelServices

router = APIRouter(prefix="/databases", tags=["databases"])


def _svc(r: Request) -> ModelServices:
    return r.app.state.services


def _tq(r: Request) -> TaskQueue:
    return r.app.state.task_queue


async def _run(svc: ModelServices, fn):
    """Run a blocking DB call in the ML executor and return the result."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(svc.executor, fn)


# ── Collection management ────────────────────────────────────────────────────

@router.get("")
async def list_databases(request: Request):
    svc = _svc(request)
    names = await _run(svc, svc.db.list_databases)
    result = []
    for n in names:
        count = await _run(svc, lambda name=n: svc.db.count(name))
        result.append({"name": n, "count": count})
    return result


@router.post("")
async def create_database(request: Request, name: str = Body(..., embed=True)):
    svc = _svc(request)
    await _run(svc, lambda: svc.db.create_database(name))
    return {"name": name, "created": True}


@router.delete("/{name}")
async def delete_database(name: str, request: Request):
    svc = _svc(request)
    await _run(svc, lambda: svc.db.delete_database(name))
    return {"name": name, "deleted": True}


# ── Facts CRUD ───────────────────────────────────────────────────────────────

@router.get("/{name}/facts")
async def list_facts(
    name: str,
    request: Request,
    limit: int = 20,
    offset: int = 0,
    query: Optional[str] = None,
):
    svc = _svc(request)
    facts = await _run(svc, lambda: svc.db.list_facts(name, limit, offset, query))
    total = await _run(svc, lambda: svc.db.count(name))
    return {"database": name, "total": total, "facts": facts}


@router.post("/{name}/facts")
async def add_facts(
    name: str,
    request: Request,
    file: Optional[UploadFile] = File(None),
    facts_text: Optional[str] = Form(None),
    tags: str = Form(""),
    source: str = Form("user"),
):
    """Add facts from a text field (semicolon-separated) or a .txt file upload.

    Goes through the task queue because embedding may take several seconds.
    Returns {task_id} for polling.
    """
    svc = _svc(request)
    task_queue = _tq(request)

    fact_list: List[str] = []

    if file and file.filename:
        content = (await file.read()).decode("utf-8", errors="replace")
        fact_list = [ln.strip() for ln in content.splitlines() if ln.strip()]
    elif facts_text:
        fact_list = [f.strip() for f in facts_text.split(";") if f.strip()]

    if not fact_list:
        return JSONResponse({"error": "No non-empty facts provided."}, status_code=400)

    def _run_add():
        ids = svc.db.add_facts(name, fact_list, source=source, tags=tags)
        return {"database": name, "added": len(ids), "ids": ids}

    task_id = await task_queue.submit(_run_add)
    return {"task_id": task_id}


@router.delete("/{name}/facts")
async def delete_facts(
    name: str,
    request: Request,
    ids: List[str] = Body(...),
):
    svc = _svc(request)
    await _run(svc, lambda: svc.db.delete_facts(name, ids))
    return {"database": name, "deleted": len(ids)}
