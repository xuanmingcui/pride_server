#!/usr/bin/env python3
"""Launch PRIDE with both Discord bot and web server sharing one set of models.

Usage:
    DISCORD_TOKEN=your_token python run_all.py [--host 0.0.0.0] [--port 8080]

Models are loaded a single time at startup, then shared between:
  • Discord bot   — slash commands (/scenegraph, /validate, …)
  • FastAPI server — HTTP API + SPA frontend at http://localhost:8080
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
from src.bot.main import PrideBot

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pride")


async def main(host: str, port: int) -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        sys.exit("ERROR: DISCORD_TOKEN environment variable not set.")

    cfg = load_config()
    log.info("Initialising model services (shared) …")
    services = await ModelServices.create(cfg)
    log.info("Models ready.")

    # FastAPI server
    app = create_app(services)
    uv_config = uvicorn.Config(app, host=host, port=port, log_level="info", loop="none")
    uv_server = uvicorn.Server(uv_config)

    # Discord bot — receives pre-loaded services, skips model loading in setup_hook
    bot = PrideBot(cfg, services=services)

    log.info("Starting Discord bot + web server on %s:%d …", host, port)
    await asyncio.gather(
        uv_server.serve(),
        bot.start(token),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRIDE — Discord + web")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
