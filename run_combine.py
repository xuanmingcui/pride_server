#!/usr/bin/env python3
"""Launch PRIDE as ONE backend doing the jobs of both run_all.py and misinfo/serve.py,
sharing a SINGLE vLLM model on the GPU.

run_all.py + src/misinfo/serve.py each load their own vLLM engine (two ~4B models on
two GPUs). This entry point loads the model ONCE (via ModelServices) and makes the
misinfo GraphCheck verifier reuse that same engine, so the whole system fits one GPU:

  • Discord bot    — slash commands (/scenegraph, /validate, …)
  • FastAPI server  — HTTP API + SPA frontend            (http://localhost:8080)
  • Misinfo backend — /api/misinfo/{health,run,verify,verify_decomposition,tasks/*}
                      served IN-PROCESS (no proxy to :8090), sharing the main vLLM

How the LLM is shared: the misinfo verifier normally builds its own
``VLLMWrapper`` (a second vLLM load). We construct a VLLMWrapper bound to the
already-loaded ``services.backend.llm`` and patch it into the misinfo code path so
``GraphCheck`` reuses it. All LLM work (scene-graph + misinfo) is funnelled through
the one single-worker executor / task queue, so the shared engine is never called
concurrently from two threads.

Note: the shared model is the main backend's model (a Qwen3-VL-* model by default,
see config.yaml `model.name`), NOT misinfo's standalone `llm_name`. The verifier's
text prompts run against it via vLLM's chat API.

Usage:
    DISCORD_TOKEN=your_token python run_combine.py [--host 0.0.0.0] [--port 8080]
"""
from __future__ import annotations

import os

# Redirect all ML caches before any torch/vllm imports (mirrors run_all.py).
for _var, _default in [
    ("HF_HOME",                 "/workspace/.cache/huggingface"),
    ("HUGGINGFACE_HUB_CACHE",   "/workspace/.cache/huggingface/hub"),
    ("TORCH_HOME",              "/workspace/.cache/torch"),
    ("TORCHINDUCTOR_CACHE_DIR", "/workspace/.cache/torch/inductor"),
    ("TRITON_CACHE_DIR",        "/workspace/.cache/triton"),
    ("VLLM_CACHE_ROOT",         "/workspace/.cache/vllm"),
    ("TMPDIR",                  "/workspace/tmp"),
]:
    os.environ.setdefault(_var, _default)
    os.makedirs(os.environ[_var], exist_ok=True)

import argparse
import asyncio
import logging
import sys
import traceback

import uvicorn
from dotenv import load_dotenv
# Imported at MODULE level (not inside the router factory) so that, under
# `from __future__ import annotations`, FastAPI's get_type_hints can resolve the
# `request: Request` annotations against module globals. If Request lives only as a
# local in the factory, FastAPI can't resolve it and treats `request` as a query
# param -> every call 422s with {"loc":["query","request"],"msg":"Field required"}.
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# Path setup: repo root for `src.*`; src/ + src/misinfo/ for the misinfo modules
# (which mix `misinfo.*` package imports with bare `graphcheck_misinfo` imports).
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
_MIS = os.path.join(_SRC, "misinfo")
for _p in (_ROOT, _SRC, _MIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.bot.config import load_config
from src.services import ModelServices
from src.api.app import create_app
from src.bot.main import PrideBot

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pride.combine")

_MISINFO_REQUIRED_KEYS = ("input_filename", "retriever_corpus_dir")


# --------------------------------------------------------------------------- #
# Shared-LLM misinfo verifier
# --------------------------------------------------------------------------- #
def _build_shared_verifier(services: ModelServices, mis_cfg: dict):
    """Build the GraphCheck verifier so it reuses the main vLLM engine.

    Runs on the single ML worker thread. Loads the pyserini retriever (CPU/JVM)
    but NOT a second LLM — it borrows ``services.backend.llm``.
    """
    from transformers import AutoTokenizer
    from misinfo.vllm_utils import VLLMWrapper
    from misinfo.graphcheck_misinfo import build_verifier, GraphCheck

    backend = services.backend
    if not hasattr(backend, "llm"):
        raise RuntimeError(
            "Shared-LLM mode requires a vLLM backend (set model.backend: vllm in "
            f"config.yaml); got {type(backend).__name__} with no .llm engine."
        )
    shared_llm = backend.llm
    model_name = backend.model_name

    # A VLLMWrapper bound to the already-loaded engine. We bypass __init__ (which
    # would load a second model) and set exactly the attributes its methods use.
    wrapper = VLLMWrapper.__new__(VLLMWrapper)
    wrapper.model_name = model_name
    wrapper.llm = shared_llm
    wrapper.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _tmpl = getattr(wrapper.tokenizer, "chat_template", None) or ""
    wrapper._supports_thinking = "enable_thinking" in _tmpl
    wrapper._think_close_token_id = wrapper._maybe_get_single_token_id("</think>")

    # Patch the name `VLLMWrapper` in the module where Direct.__init__ resolves it,
    # so `self.llm_model = VLLMWrapper(args.llm_name)` returns our shared wrapper.
    # GraphCheck.__mro__[1] is Direct; its module is the exact namespace to patch
    # (robust whether imported as `direct_misinfo` or `misinfo.direct_misinfo`).
    direct_cls = GraphCheck.__mro__[1]
    direct_mod = sys.modules[direct_cls.__module__]
    _orig = getattr(direct_mod, "VLLMWrapper", None)
    direct_mod.VLLMWrapper = lambda *a, **k: wrapper
    try:
        verifier = build_verifier(
            dataset=mis_cfg.get("dataset", "custom"),
            input_filename=mis_cfg["input_filename"],
            retriever_corpus_dir=mis_cfg["retriever_corpus_dir"],
            llm_name=model_name,  # logged only; the wrapper above is what's used
            setting=mis_cfg.get("setting", "open-book"),
            mode=mis_cfg.get("mode", "weight_update"),
            update_mode=mis_cfg.get("update_mode", "weighted"),
            top_k=int(mis_cfg.get("top_k", 10)),
            path_limit=int(mis_cfg.get("path_limit", 5)),
        )
    finally:
        direct_mod.VLLMWrapper = _orig
    return verifier


# --------------------------------------------------------------------------- #
# In-process misinfo router (mirrors src/misinfo/serve.py's endpoints, but uses
# the shared verifier + the app's single task queue / executor).
# --------------------------------------------------------------------------- #
def _make_misinfo_router(services: ModelServices):
    router = APIRouter(prefix="/misinfo", tags=["misinfo"])

    def _not_loaded(request: Request) -> JSONResponse:
        return JSONResponse(
            {"error": "verifier not loaded", "load_error": request.app.state.misinfo_load_error},
            status_code=503,
        )

    async def _body(request: Request) -> dict:
        try:
            return await request.json() or {}
        except Exception:
            return {}

    @router.get("/health")
    async def health(request: Request):
        return {
            "status": "ok",
            "model_loaded": request.app.state.misinfo_verifier is not None,
            "load_error": request.app.state.misinfo_load_error,
            "config": request.app.state.misinfo_cfg,
            "shared_llm": getattr(services.backend, "model_name", None),
        }

    @router.post("/run")
    async def run(request: Request):
        verifier = request.app.state.misinfo_verifier
        if verifier is None:
            return _not_loaded(request)
        from misinfo.graphcheck_misinfo import run_graphcheck_job
        mode = request.app.state.misinfo_cfg.get("mode", "weight_update")
        # Reuse the app's task queue -> same single worker as scene-graph jobs.
        task_id = await request.app.state.task_queue.submit(
            lambda: run_graphcheck_job(verifier, mode=mode)
        )
        return {"task_id": task_id}

    @router.post("/verify")
    async def verify(request: Request):
        verifier = request.app.state.misinfo_verifier
        if verifier is None:
            return _not_loaded(request)
        body = await _body(request)
        prompt = body.get("prompt", "")
        prompt = prompt.strip() if isinstance(prompt, str) else ""
        if not prompt:
            return JSONResponse({"error": "Provide a non-empty 'prompt'."}, status_code=400)
        top_k = body.get("top_k")
        try:
            result = await services.run_in_thread(
                lambda: verifier.single_prompt(prompt, top_k=top_k)
            )
        except Exception as exc:
            log.error("single_prompt failed: %s\n%s", exc, traceback.format_exc())
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
        return result

    @router.post("/verify_decomposition")
    async def verify_decomposition(request: Request):
        verifier = request.app.state.misinfo_verifier
        if verifier is None:
            return _not_loaded(request)
        body = await _body(request)
        prompt = body.get("prompt", "")
        prompt = prompt.strip() if isinstance(prompt, str) else ""
        if not prompt:
            return JSONResponse({"error": "Provide a non-empty 'prompt'."}, status_code=400)
        top_k = body.get("top_k")
        try:
            result = await services.run_in_thread(
                lambda: verifier.single_prompt_decomposition(prompt, top_k=top_k)
            )
        except Exception as exc:
            log.error("single_prompt_decomposition failed: %s\n%s", exc, traceback.format_exc())
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
        return result

    @router.get("/tasks/{task_id}")
    async def task_status(task_id: str, request: Request):
        info = request.app.state.task_queue.get(task_id)
        if info is None:
            return JSONResponse({"error": "unknown task_id"}, status_code=404)
        return info.to_dict()

    return router


async def main(host: str, port: int) -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        sys.exit("ERROR: DISCORD_TOKEN environment variable not set.")

    cfg = load_config()
    log.info("Initialising model services (shared vLLM engine) …")
    services = await ModelServices.create(cfg)
    log.info("Models ready: %s", cfg["model"]["name"])

    # Misinfo verifier — reuse the shared engine. Mirrors serve.py's graceful
    # degradation: if config is incomplete or the build fails, the misinfo
    # endpoints return 503 with the reason instead of taking the whole app down.
    from misinfo.serve import load_misinfo_config
    mis_cfg = load_misinfo_config()
    verifier = None
    load_error = None
    missing = [k for k in _MISINFO_REQUIRED_KEYS if not mis_cfg.get(k)]
    if missing:
        load_error = (
            f"misinfo_config.yaml missing {missing} — misinfo endpoints disabled. "
            f"Fill them in (or set MISINFO_* env vars) and restart."
        )
        log.warning(load_error)
    else:
        try:
            log.info("Building misinfo verifier (pyserini retriever; reuses the vLLM engine) …")
            verifier = await services.run_in_thread(_build_shared_verifier, services, mis_cfg)
            log.info("Misinfo verifier ready — sharing the main vLLM model.")
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
            log.error("Misinfo verifier build failed: %s\n%s", exc, traceback.format_exc())

    # FastAPI server — inject the in-process misinfo router (no proxy to :8090).
    app = create_app(services, misinfo_router=_make_misinfo_router(services))
    app.state.misinfo_verifier = verifier
    app.state.misinfo_cfg = mis_cfg
    app.state.misinfo_load_error = load_error

    uv_config = uvicorn.Config(app, host=host, port=port, log_level="info", loop="none")
    uv_server = uvicorn.Server(uv_config)

    # Discord bot — receives pre-loaded services, skips model loading in setup_hook.
    bot = PrideBot(cfg, services=services)

    log.info("Starting combined backend (Discord + web + misinfo, one shared LLM) on %s:%d …",
             host, port)
    await asyncio.gather(
        uv_server.serve(),
        bot.start(token),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRIDE — combined backend (one shared LLM)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
