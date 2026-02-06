"""FastAPI routes."""

import asyncio
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import APIRouter, Request, Form, Query, BackgroundTasks
from fastapi.responses import HTMLResponse

from ..db.models import (
    get_document,
    get_document_count,
    get_documents_by_status,
    get_all_documents,
    get_image,
    get_images_for_document,
    search_images,
    get_image_count,
    get_all_images,
    get_entities_by_type,
    get_entities_for_document,
    create_document,
    update_document_text,
    update_document_status,
    get_duplicate_groups,
)
from ..search import keyword_search, semantic_search, hybrid_search
from ..detective import find_connections, find_co_occurring_entities, timeline_search
from ..detective.timeline import parse_date
from ..config import DATA_DIR

router = APIRouter()


# ============== Indexing State ==============

@dataclass
class IndexingJob:
    """Track indexing job state."""
    id: str
    status: str = "idle"  # idle, running, completed, failed
    mode: str = "embedded"
    total: int = 0
    processed: int = 0
    failed: int = 0
    current_doc: str = ""
    started_at: datetime = None
    completed_at: datetime = None
    error: str = ""
    log: list = field(default_factory=list)


# Global indexing state
_indexing_job = IndexingJob(id="main")
_indexing_lock = threading.Lock()


def get_indexing_job() -> IndexingJob:
    """Get current indexing job state."""
    return _indexing_job


def run_indexing_job(mode: str, limit: int):
    """Run indexing in background thread."""
    global _indexing_job

    with _indexing_lock:
        if _indexing_job.status == "running":
            return

        _indexing_job.status = "running"
        _indexing_job.mode = mode
        _indexing_job.processed = 0
        _indexing_job.failed = 0
        _indexing_job.current_doc = ""
        _indexing_job.started_at = datetime.now()
        _indexing_job.completed_at = None
        _indexing_job.error = ""
        _indexing_job.log = []

    def _run():
        global _indexing_job
        try:
            _do_indexing(mode, limit)
            _indexing_job.status = "completed"
        except Exception as e:
            _indexing_job.status = "failed"
            _indexing_job.error = str(e)
        finally:
            _indexing_job.completed_at = datetime.now()
            _indexing_job.current_doc = ""

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _do_indexing(mode: str, limit: int):
    """Actual indexing logic."""
    global _indexing_job

    from ..extraction import (
        extract_text_from_pdf, extract_embedded_text, extract_hybrid_text,
        classify_file, ALL_EXTENSIONS, EXTRACTORS,
    )
    from ..indexing import chunk_document, VectorStore, get_embeddings
    from ..db.models import get_chunks_for_document, update_chunk_embedding

    if not DATA_DIR.exists():
        _indexing_job.log.append("Data directory not found")
        return

    # Find all supported files (not just PDFs)
    seen_lower: set[str] = set()
    all_files: list[Path] = []
    for f in DATA_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS:
            key = str(f).lower()
            if key not in seen_lower:
                seen_lower.add(key)
                all_files.append(f)

    all_files = all_files[:limit]
    _indexing_job.total = len(all_files)
    _indexing_job.log.append(f"Found {len(all_files)} files")

    # Register documents
    registered = 0
    for file_path in all_files:
        doc_id = file_path.stem
        vol = file_path.parent.name if file_path.parent != DATA_DIR else ""

        existing = get_document(doc_id)
        if not existing:
            ft = classify_file(file_path)
            create_document(doc_id=doc_id, file_path=str(file_path), volume=vol, file_type=ft)
            registered += 1

    _indexing_job.log.append(f"Registered {registered} new documents")

    if mode == "register":
        _indexing_job.processed = len(all_files)
        return

    # Process documents
    pending = get_documents_by_status("pending", limit=limit)
    _indexing_job.total = len(pending)

    if not pending:
        _indexing_job.log.append("No pending documents")
        return

    vector_store = VectorStore()

    for doc in pending:
        doc_id = doc["doc_id"]
        file_path = doc["file_path"]
        doc_file_type = doc.get("file_type", "pdf")
        _indexing_job.current_doc = doc_id

        try:
            # Non-PDF types: use extractor registry
            if doc_file_type in EXTRACTORS:
                extractor = EXTRACTORS[doc_file_type]
                text, metadata = extractor(file_path)

                if not text.strip():
                    _indexing_job.processed += 1
                    continue

                update_document_text(
                    doc_id, text,
                    duration_seconds=metadata.get("duration_seconds"),
                )
            else:
                # PDF: use existing mode logic
                embedded_text = None
                ocr_text = None

                if mode == "embedded":
                    text, page_count = extract_embedded_text(file_path)
                    if not text.strip():
                        _indexing_job.processed += 1
                        continue
                    embedded_text = text
                elif mode == "hybrid":
                    text, embedded_text, ocr_text, page_count = extract_hybrid_text(file_path)
                    if not text.strip():
                        _indexing_job.processed += 1
                        continue
                else:  # ocr
                    text, page_count = extract_text_from_pdf(file_path)
                    ocr_text = text

                update_document_text(
                    doc_id, text, page_count,
                    embedded_text=embedded_text,
                    ocr_text=ocr_text,
                )

            # Chunk and embed
            chunk_ids = chunk_document(doc_id)

            if chunk_ids:
                doc_refreshed = get_document(doc_id)
                chunks = get_chunks_for_document(doc_refreshed["id"])

                chunk_texts = [c["text"] for c in chunks]
                embeddings = get_embeddings(chunk_texts)

                ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [
                    {"document_id": str(doc_refreshed["id"]), "chunk_index": i, "doc_id": doc_id}
                    for i in range(len(chunks))
                ]
                vector_store.add(ids, embeddings, chunk_texts, metadatas)

                for chunk, emb_id in zip(chunks, ids):
                    update_chunk_embedding(chunk["id"], emb_id)

            _indexing_job.processed += 1

        except Exception as e:
            _indexing_job.failed += 1
            _indexing_job.log.append(f"Error processing {doc_id}: {str(e)[:100]}")
            update_document_status(doc_id, "failed")


def get_templates(request: Request):
    """Get templates from app state."""
    return request.app.state.templates


# ============== Pages ==============

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search."""
    templates = get_templates(request)

    doc_count = get_document_count()
    img_count = get_image_count()
    completed = len(get_documents_by_status("completed", limit=100000))

    return templates.TemplateResponse("index.html", {
        "request": request,
        "doc_count": doc_count,
        "img_count": img_count,
        "completed": completed,
    })


@router.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request, page: int = 1, per_page: int = 50):
    """Browse all documents."""
    templates = get_templates(request)

    offset = (page - 1) * per_page
    docs = get_all_documents(limit=per_page)  # TODO: Add offset support
    total = get_document_count()

    return templates.TemplateResponse("documents.html", {
        "request": request,
        "documents": docs,
        "page": page,
        "per_page": per_page,
        "total": total,
    })


@router.get("/document/{doc_id}", response_class=HTMLResponse)
async def document_detail(request: Request, doc_id: str):
    """View a single document."""
    templates = get_templates(request)

    doc = get_document(doc_id)
    if not doc:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": f"Document not found: {doc_id}"
        })

    images = get_images_for_document(doc["id"])
    entities = get_entities_for_document(doc["id"])

    return templates.TemplateResponse("document.html", {
        "request": request,
        "doc": doc,
        "images": images,
        "entities": entities,
        "config": {"DATA_DIR": str(DATA_DIR)},
    })


@router.get("/images", response_class=HTMLResponse)
async def images_page(request: Request, page: int = 1):
    """Browse all images."""
    templates = get_templates(request)

    per_page = 24
    images = get_all_images(limit=per_page)  # TODO: pagination
    total = get_image_count()

    return templates.TemplateResponse("images.html", {
        "request": request,
        "images": images,
        "page": page,
        "total": total,
    })


@router.get("/entities", response_class=HTMLResponse)
async def entities_page(request: Request, entity_type: str = "PERSON"):
    """View top entities."""
    templates = get_templates(request)

    entities = get_entities_by_type(entity_type, limit=100)

    return templates.TemplateResponse("entities.html", {
        "request": request,
        "entities": entities,
        "entity_type": entity_type,
    })


@router.get("/timeline", response_class=HTMLResponse)
async def timeline_page(request: Request, from_date: str = None, to_date: str = None):
    """Timeline view."""
    templates = get_templates(request)

    start = parse_date(from_date) if from_date else None
    end = parse_date(to_date) if to_date else None

    results = timeline_search(start_date=start, end_date=end, limit=100)

    return templates.TemplateResponse("timeline.html", {
        "request": request,
        "results": results,
        "from_date": from_date or "",
        "to_date": to_date or "",
    })


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """RAG chat interface."""
    templates = get_templates(request)
    return templates.TemplateResponse("chat.html", {"request": request})


@router.get("/indexing", response_class=HTMLResponse)
async def indexing_page(request: Request):
    """Indexing management page."""
    templates = get_templates(request)

    job = get_indexing_job()
    doc_count = get_document_count()
    pending = len(get_documents_by_status("pending", limit=100000))
    completed = len(get_documents_by_status("completed", limit=100000))
    failed = len(get_documents_by_status("failed", limit=100000))

    # Count all supported files in data directory
    from ..extraction import ALL_EXTENSIONS
    file_count = 0
    if DATA_DIR.exists():
        for f in DATA_DIR.rglob("*"):
            if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS:
                file_count += 1

    return templates.TemplateResponse("indexing.html", {
        "request": request,
        "job": job,
        "doc_count": doc_count,
        "pdf_count": file_count,
        "pending": pending,
        "completed": completed,
        "failed": failed,
    })


@router.get("/duplicates", response_class=HTMLResponse)
async def duplicates_page(request: Request):
    """View duplicate file groups."""
    templates = get_templates(request)

    groups = get_duplicate_groups()
    total_dupes = sum(g["count"] - 1 for g in groups)

    return templates.TemplateResponse("duplicates.html", {
        "request": request,
        "groups": groups,
        "total_dupes": total_dupes,
    })


# ============== HTMX Partials ==============

@router.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    search_type: str = Form("hybrid"),
    limit: int = Form(20)
):
    """Search documents (HTMX partial)."""
    templates = get_templates(request)

    if not query.strip():
        return templates.TemplateResponse("partials/search_results.html", {
            "request": request,
            "results": [],
            "query": "",
        })

    if search_type == "semantic":
        results = semantic_search(query, limit=limit)
    elif search_type == "keyword":
        results = keyword_search(query, limit=limit)
    else:
        results = hybrid_search(query, limit=limit)

    return templates.TemplateResponse("partials/search_results.html", {
        "request": request,
        "results": results,
        "query": query,
        "search_type": search_type,
    })


@router.post("/search-images", response_class=HTMLResponse)
async def search_images_partial(
    request: Request,
    query: str = Form(...),
    limit: int = Form(20)
):
    """Search images (HTMX partial)."""
    templates = get_templates(request)

    if not query.strip():
        return templates.TemplateResponse("partials/image_results.html", {
            "request": request,
            "images": [],
        })

    images = search_images(query, limit=limit)

    return templates.TemplateResponse("partials/image_results.html", {
        "request": request,
        "images": images,
        "query": query,
    })


@router.post("/ask", response_class=HTMLResponse)
async def ask_question(
    request: Request,
    question: str = Form(...)
):
    """Ask a question using RAG (HTMX partial)."""
    templates = get_templates(request)

    if not question.strip():
        return templates.TemplateResponse("partials/chat_response.html", {
            "request": request,
            "answer": "",
            "sources": [],
        })

    from ..llm import ask_with_sources

    result = ask_with_sources(question)

    return templates.TemplateResponse("partials/chat_response.html", {
        "request": request,
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
    })


@router.get("/connections/{entity}", response_class=HTMLResponse)
async def connections_partial(request: Request, entity: str):
    """Get entity connections (HTMX partial)."""
    templates = get_templates(request)

    connections = find_connections(entity)
    co_people = find_co_occurring_entities(entity, entity_type="PERSON", limit=20)

    return templates.TemplateResponse("partials/connections.html", {
        "request": request,
        "entity": entity,
        "connections": connections,
        "co_people": co_people,
    })


@router.post("/indexing/start", response_class=HTMLResponse)
async def start_indexing(
    request: Request,
    mode: str = Form("embedded"),
    limit: int = Form(1000),
    background_tasks: BackgroundTasks = None
):
    """Start indexing job (HTMX)."""
    templates = get_templates(request)

    job = get_indexing_job()
    if job.status == "running":
        return templates.TemplateResponse("partials/indexing_status.html", {
            "request": request,
            "job": job,
            "message": "Indexing already in progress",
        })

    # Start indexing in background
    run_indexing_job(mode, limit)

    job = get_indexing_job()
    return templates.TemplateResponse("partials/indexing_status.html", {
        "request": request,
        "job": job,
        "message": "Indexing started",
    })


@router.get("/indexing/status", response_class=HTMLResponse)
async def indexing_status(request: Request):
    """Get indexing status (HTMX polling)."""
    templates = get_templates(request)

    job = get_indexing_job()
    pending = len(get_documents_by_status("pending", limit=100000))
    completed = len(get_documents_by_status("completed", limit=100000))
    failed = len(get_documents_by_status("failed", limit=100000))

    return templates.TemplateResponse("partials/indexing_status.html", {
        "request": request,
        "job": job,
        "pending": pending,
        "completed": completed,
        "failed": failed,
    })


# ============== API Endpoints ==============

@router.get("/api/status")
async def api_status():
    """Get system status."""
    return {
        "documents": get_document_count(),
        "images": get_image_count(),
        "completed": len(get_documents_by_status("completed", limit=100000)),
        "pending": len(get_documents_by_status("pending", limit=100000)),
    }


@router.get("/api/search")
async def api_search(
    q: str,
    type: str = "hybrid",
    limit: int = 20
):
    """Search API endpoint."""
    if type == "semantic":
        results = semantic_search(q, limit=limit)
    elif type == "keyword":
        results = keyword_search(q, limit=limit)
    else:
        results = hybrid_search(q, limit=limit)

    return {"query": q, "results": results}
