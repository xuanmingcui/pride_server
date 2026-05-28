#!/usr/bin/env python3
"""Launch the PRIDE web server only (no Discord bot).

Usage:
    python run_web.py [--host 0.0.0.0] [--port 8080]

Models are loaded once at startup, then the FastAPI server handles all
requests via /api/* endpoints.  The frontend SPA is served at /.
"""
from __future__ import annotations

import os

# Redirect all ML caches before any torch/vllm imports.
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

import uvicorn
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from src.bot.config import load_config
from src.services import ModelServices
from src.api.app import create_app

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pride.web")


async def main(host: str, port: int) -> None:
    cfg = load_config()
    log.info("Initialising model services …")
    services = await ModelServices.create(cfg)
    log.info("Models ready. Starting web server on %s:%d …", host, port)

    app = create_app(services)
    config = uvicorn.Config(app, host=host, port=port, log_level="info", loop="none")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRIDE web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
