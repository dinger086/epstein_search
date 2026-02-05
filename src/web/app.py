"""FastAPI application."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import DB_DIR, ensure_dirs
from ..db import init_db

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def basename_filter(path: str) -> str:
    """Jinja2 filter to get basename from path."""
    if not path:
        return ""
    return os.path.basename(path)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    ensure_dirs()
    init_db()

    app = FastAPI(
        title="Detective Document Search",
        description="Search and analyze documents with AI",
        version="0.1.0"
    )

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # Add custom filters
    templates.env.filters["basename"] = basename_filter

    app.state.templates = templates

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Mount images directory for serving extracted images
    images_dir = DB_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    # Mount data directory for serving PDF files
    from ..config import DATA_DIR
    if DATA_DIR.exists():
        app.mount("/pdfs", StaticFiles(directory=str(DATA_DIR)), name="pdfs")

    # Import and include routes
    from .routes import router
    app.include_router(router)

    return app


# Create app instance for uvicorn
app = create_app()
