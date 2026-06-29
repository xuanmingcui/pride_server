"""Proxy router: forwards /api/misinfo/* to the standalone misinfo service.

The misinfo GraphCheck backend runs in its own conda env / process (see
src/misinfo/serve.py, default http://127.0.0.1:8090) because its deps
(vllm + pyserini + numpy 1.26) are incompatible with run_all.py's env. This
router lets the single-origin SPA reach it without CORS or a second base URL.

Override the upstream with the MISINFO_UPSTREAM env var.
"""
from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/misinfo", tags=["misinfo"])

_UPSTREAM = os.environ.get("MISINFO_UPSTREAM", "http://127.0.0.1:8090")
# Long timeout: a full-dataset /run kicks off async (returns a task_id fast), but
# keep generous headroom for slower calls / cold upstream.
_client = httpx.AsyncClient(base_url=_UPSTREAM, timeout=httpx.Timeout(120.0))

_HOP_BY_HOP = {"content-encoding", "transfer-encoding", "content-length", "connection"}


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    body = await request.body()
    try:
        upstream = await _client.request(
            request.method,
            f"/{path}",
            content=body,
            params=request.query_params,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
    except httpx.ConnectError:
        return JSONResponse(
            {"error": f"misinfo service unreachable at {_UPSTREAM}. "
                      f"Start it: `conda activate misinfo && CUDA_VISIBLE_DEVICES=1 python -m misinfo.serve`"},
            status_code=502,
        )
    except httpx.TimeoutException:
        return JSONResponse({"error": "misinfo service timed out"}, status_code=504)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP},
    )
