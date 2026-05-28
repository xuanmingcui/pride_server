"""FastAPI application factory for the PRIDE web API.

Usage:
    services = await ModelServices.create(cfg)
    app = create_app(services)
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.services import ModelServices
from src.api.task_queue import TaskQueue
from src.api.routes import scenegraph as sg_routes
from src.api.routes import validation as val_routes
from src.api.routes import database as db_routes
from src.api.routes import tasks as task_routes
from src.api.routes import prompts as prompt_routes

_FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


def create_app(services: ModelServices) -> FastAPI:
    app = FastAPI(
        title="PRIDE API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    task_queue = TaskQueue(services.executor)
    app.state.services = services
    app.state.task_queue = task_queue

    @app.on_event("startup")
    async def _startup():
        await task_queue.start()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sg_routes.router, prefix="/api")
    app.include_router(val_routes.router, prefix="/api")
    app.include_router(db_routes.router, prefix="/api")
    app.include_router(task_routes.router, prefix="/api")
    app.include_router(prompt_routes.router, prefix="/api")

    if _FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

    return app
