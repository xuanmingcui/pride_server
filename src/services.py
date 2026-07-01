"""Shared ML services — loaded once and used by both the Discord bot and the web API.

Extracting model initialization here lets run.py create one set of model
instances and hand them to both the Discord bot and the FastAPI server, avoiding
double-loading the multi-GB GPU models.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.core.mllm import BaseMLLM
    from src.core.embedder import MultimodalEmbedder
    from src.core.scenegraph import SceneGraphPipeline
    from src.core.prompts import PromptStore
    from src.core.video_index import VideoIndex
    from src.validation.database import FactDatabase
    from src.validation.pipeline import ValidationPipeline

log = logging.getLogger("pride.services")


@dataclass
class ModelServices:
    """Container for all loaded ML models and pipelines shared across interfaces."""
    cfg: Dict[str, Any]
    executor: ThreadPoolExecutor
    backend: "BaseMLLM"
    sg_pipeline: "SceneGraphPipeline"
    embedder: "MultimodalEmbedder"
    db: "FactDatabase"
    val_pipeline: "ValidationPipeline"
    prompt_store: "PromptStore"
    video_index: "VideoIndex"

    @classmethod
    async def create(cls, cfg: Dict[str, Any]) -> "ModelServices":
        """Initialize all models and return a ready ModelServices instance."""
        from src.core.gpu_utils import assign_gpus, whisper_compute_type
        from src.core.mllm import get_backend
        from src.core.embedder import get_embedder
        from src.core.prompts import init_store
        from src.core.scenegraph import SceneGraphPipeline
        from src.core.video_index import VideoIndex
        from src.validation.database import FactDatabase
        from src.validation.pipeline import ValidationPipeline

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pride-ml")
        loop = asyncio.get_event_loop()

        gpu = assign_gpus(cfg["model"])
        whisper_cfg = dict(cfg["whisper"])
        whisper_cfg["device"] = gpu["whisper"]
        whisper_cfg["compute_type"] = whisper_compute_type(
            gpu["whisper"], preferred=whisper_cfg.get("compute_type", "float16")
        )
        log.info(
            "Device assignment — MLLM: %s | Whisper: %s | Embedding: %s",
            gpu["mllm"], gpu["whisper"], gpu["embed"],
        )

        log.info("Loading MLLM backend …")
        backend = await loop.run_in_executor(executor, lambda: get_backend(cfg["model"]))
        log.info("Backend loaded: %s", cfg["model"]["name"])

        paths = cfg["paths"]
        sg_cfg = dict(cfg["scenegraph"])
        sg_cfg.setdefault("max_model_len", cfg["model"].get("max_model_len", 32768))
        sg_pipeline = SceneGraphPipeline(
            backend=backend,
            whisper_config=whisper_cfg,
            scenegraph_config=sg_cfg,
            tmp_dir=paths["tmp_dir"],
        )

        val_cfg = dict(cfg["validation"])
        val_cfg["embed_device"] = gpu["embed"]
        log.info("Loading embedding model …")
        embedder = await loop.run_in_executor(executor, lambda: get_embedder(val_cfg))
        log.info("Embedding model loaded.")

        prompt_store = init_store(paths["db_dir"])

        db = FactDatabase(db_path=paths["db_dir"], embedder=embedder)

        import os
        video_index_dir = paths.get(
            "video_index_dir", os.path.join(paths["db_dir"], "video_index")
        )
        video_index = VideoIndex(index_path=video_index_dir, embedder=embedder)
        val_pipeline = ValidationPipeline(
            backend=backend,
            db=db,
            whisper_config=whisper_cfg,
            top_k=cfg["validation"]["top_k"],
            fps=float(cfg["validation"].get("fps", 1.0)),
            min_frames=int(cfg["validation"].get("min_frames", 4)),
            tmp_dir=paths["tmp_dir"],
        )

        return cls(
            cfg=cfg,
            executor=executor,
            backend=backend,
            sg_pipeline=sg_pipeline,
            embedder=embedder,
            db=db,
            val_pipeline=val_pipeline,
            prompt_store=prompt_store,
            video_index=video_index,
        )

    async def run_in_thread(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: fn(*args, **kwargs))
