"""Discord bot entry point for the PRIDE server.

Loads models once at startup, then serves all slash commands.
Long-running ML tasks are dispatched to a single-worker ThreadPoolExecutor
so the Discord event loop is never blocked.

Run:
    DISCORD_TOKEN=your_token python -m src.bot.main
Or:
    python src/bot/main.py
"""
from __future__ import annotations

import os

# Redirect all ML caches to /workspace before any torch/vllm imports.
# Must be set here so vLLM's EngineCore subprocess inherits them at spawn time.
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

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Add project root to path so relative imports work when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.bot.config import load_config
from src.bot.commands.help_cmd import HelpCog
from src.bot.commands.prompts_cmd import PromptsCog
from src.bot.commands.scenegraph_cmd import SceneGraphCog
from src.bot.commands.validate_cmd import ValidateCog

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pride")


class PrideBot(commands.Bot):
    def __init__(self, cfg: dict, services=None):
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)
        self.cfg = cfg
        self._services = services  # pre-loaded ModelServices (optional)

        # Placeholders — populated in setup_hook from services
        self.executor    = None
        self.backend     = None
        self.sg_pipeline = None
        self.embedder    = None
        self.db          = None
        self.val_pipeline = None

    async def setup_hook(self) -> None:
        if self._services is not None:
            svc = self._services
            log.info("Using pre-loaded model services (shared with web API).")
        else:
            log.info("Loading model services for Discord-only mode …")
            from src.services import ModelServices
            svc = await ModelServices.create(self.cfg)

        self.executor    = svc.executor
        self.backend     = svc.backend
        self.sg_pipeline = svc.sg_pipeline
        self.embedder    = svc.embedder
        self.db          = svc.db
        self.val_pipeline = svc.val_pipeline

        # Register cogs
        await self.add_cog(HelpCog(self))
        await self.add_cog(SceneGraphCog(self))
        await self.add_cog(ValidateCog(self))
        await self.add_cog(PromptsCog(self))

        guild_id = self.cfg.get("discord", {}).get("guild_id")
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("Slash commands synced to guild %s (instant).", guild_id)
        else:
            await self.tree.sync()
            log.info("Slash commands synced globally (may take up to 1 hour to appear).")

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id)

    async def run_in_thread(self, fn, *args, **kwargs):
        """Run a blocking function in the ML thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: fn(*args, **kwargs))


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        sys.exit("ERROR: DISCORD_TOKEN environment variable not set.")

    cfg = load_config()
    client = PrideBot(cfg)
    client.run(token)


if __name__ == "__main__":
    main()
