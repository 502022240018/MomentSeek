from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import (
    color_grading_routes,
    entity_routes,
    folder_routes,
    job_routes,
    planner_lab_routes,
    search_routes,
    speaker_routes,
    system_routes,
    video_routes,
)
from app.platform import context


@asynccontextmanager
async def lifespan(_: FastAPI):
    context.settings.ensure_dirs()
    context.catalog.recover_color_grading_finalizations()
    if context.settings.search_prewarm_enabled:
        await asyncio.to_thread(context.search_engine.prewarm)
    context.start_indexer_daemon_if_configured()
    try:
        yield
    finally:
        context.stop_indexer_daemon()
        context.search_engine.close()


app = FastAPI(
    title="MomentSeek API",
    version=__version__,
    description="Local-first face, visual, ASR and OCR video moment retrieval MVP.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
for route_module in (
    system_routes,
    video_routes,
    speaker_routes,
    job_routes,
    entity_routes,
    folder_routes,
    search_routes,
    planner_lab_routes,
    color_grading_routes,
):
    app.include_router(route_module.router)


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        candidate = static_dir / path
        if path and candidate.is_file() and static_dir in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(static_dir / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return JSONResponse({"name": "MomentSeek", "docs": "/docs", "status": "frontend not built"})
