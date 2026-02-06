"""CLI entry point for the Detective Document Search System."""

import sys
from pathlib import Path
from datetime import date

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.markdown import Markdown

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, ensure_dirs
from src.db import init_db
from src.db.models import (
    create_document,
    get_document,
    get_documents_by_status,
    get_document_count,
    get_all_documents,
    iter_documents_by_status,
    count_documents_by_status,
    count_documents_by_file_type,
)

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Detective Document Search System - Search and analyze documents with AI."""
    ensure_dirs()
    init_db()


@cli.command()
@click.option("--limit", "-l", default=None, type=int, help="Maximum documents to process (default: no limit)")
@click.option("--volume", "-v", help="Specific volume to process")
@click.option("--mode", "-m", type=click.Choice(["ocr", "embedded", "hybrid", "register"]), default="hybrid",
              help="Processing mode: ocr (full OCR), embedded (extract text only), hybrid (smart), register (no extraction)")
@click.option("--file-type", "-t", "file_type_filter",
              type=click.Choice(["all", "pdf", "spreadsheet", "video", "audio", "litigation"]),
              default="all", help="Filter by file type")
def index(limit: int | None, volume: str, mode: str, file_type_filter: str):
    """Index documents from the data directory.

    Modes:
      - ocr: Full OCR processing with glm-ocr (slow, for scanned docs)
      - embedded: Extract embedded text only (fast, for digital PDFs)
      - hybrid: Smart mode — embedded text first, OCR only pages with poor text (recommended)
      - register: Just register documents, no text extraction
    """
    from src.extraction import (
        extract_text_from_pdf, extract_embedded_text, get_pdf_page_count,
        extract_hybrid_text, classify_file, ALL_EXTENSIONS, EXTRACTORS,
    )
    from src.indexing import chunk_document, VectorStore
    from src.db.models import update_document_text

    mode_desc = {
        "ocr": "OCR processing (slow)",
        "embedded": "Embedded text extraction (fast)",
        "hybrid": "Hybrid extraction (smart — OCR only where needed)",
        "register": "Registration only",
    }
    type_label = f" [file type: {file_type_filter}]" if file_type_filter != "all" else ""
    console.print(Panel(f"Starting document indexing - {mode_desc[mode]}{type_label}", title="Index"))

    # Find files (multi-extension scan)
    data_path = DATA_DIR
    if not data_path.exists():
        console.print(f"[red]Data directory not found: {data_path}[/red]")
        console.print("Please create the data directory and add your documents.")
        return

    # Collect all supported files, deduplicate case-insensitively
    seen_lower: set[str] = set()
    all_files: list[Path] = []
    for f in data_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS:
            key = str(f).lower()
            if key not in seen_lower:
                seen_lower.add(key)
                # Apply file type filter
                ft = classify_file(f)
                if file_type_filter == "all" or ft == file_type_filter:
                    all_files.append(f)

    if limit:
        all_files = all_files[:limit]

    if not all_files:
        console.print("[yellow]No supported files found in data directory.[/yellow]")
        return

    # Show counts by type
    type_counts: dict[str, int] = {}
    for f in all_files:
        ft = classify_file(f)
        type_counts[ft] = type_counts.get(ft, 0) + 1
    for ft, count in sorted(type_counts.items()):
        console.print(f"  {ft}: {count} files")
    console.print(f"Found {len(all_files)} files total")

    # Register documents
    registered = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Registering documents...", total=len(all_files))

        for file_path in all_files:
            doc_id = file_path.stem
            vol = file_path.parent.name if file_path.parent != data_path else ""

            if volume and vol != volume:
                progress.advance(task)
                continue

            existing = get_document(doc_id)
            if not existing:
                ft = classify_file(file_path)
                create_document(
                    doc_id=doc_id,
                    file_path=str(file_path),
                    volume=vol,
                    file_type=ft,
                )
                registered += 1

            progress.advance(task)

    console.print(f"[green]Registered {registered} new documents[/green]")

    if mode == "register":
        console.print("[yellow]Registration complete. No text extraction.[/yellow]")
        return

    # Process pending documents
    pending_count = count_documents_by_status("pending")
    if limit:
        pending_count = min(pending_count, limit)

    if pending_count == 0:
        console.print("[green]No pending documents to process[/green]")
        return

    console.print(f"\nProcessing {pending_count} documents ({mode} mode)...")

    vector_store = VectorStore()
    processed = 0
    skipped = 0
    doc_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=pending_count)

        for doc in iter_documents_by_status("pending", batch_size=500):
            if limit and doc_count >= limit:
                break
            doc_count += 1
            doc_id = doc["doc_id"]
            file_path = doc["file_path"]
            doc_file_type = doc.get("file_type", "pdf")

            # Apply file type filter during processing
            if file_type_filter != "all" and doc_file_type != file_type_filter:
                progress.advance(task)
                continue

            progress.update(task, description=f"Processing {doc_id}...")

            try:
                # Non-PDF types: use the extractor registry
                if doc_file_type in EXTRACTORS:
                    extractor = EXTRACTORS[doc_file_type]
                    text, metadata = extractor(file_path)

                    if not text.strip():
                        skipped += 1
                        progress.advance(task)
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
                            skipped += 1
                            progress.advance(task)
                            continue
                        embedded_text = text
                    elif mode == "hybrid":
                        text, embedded_text, ocr_text, page_count = extract_hybrid_text(file_path)
                        if not text.strip():
                            skipped += 1
                            progress.advance(task)
                            continue
                    else:  # ocr mode
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
                    from src.db.models import get_chunks_for_document, update_chunk_embedding
                    from src.indexing import get_embeddings
                    import gc

                    chunks = get_chunks_for_document(doc["id"])

                    # Process in batches to avoid memory issues
                    EMBED_BATCH_SIZE = 200
                    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
                        batch_end = min(batch_start + EMBED_BATCH_SIZE, len(chunks))
                        batch_chunks = chunks[batch_start:batch_end]

                        batch_texts = [c["text"] for c in batch_chunks]
                        batch_embeddings = get_embeddings(batch_texts)

                        batch_ids = [f"{doc_id}_chunk_{batch_start + i}" for i in range(len(batch_chunks))]
                        batch_metadatas = [
                            {
                                "document_id": str(doc["id"]),
                                "chunk_index": batch_start + i,
                                "doc_id": doc_id,
                                "page_number": c.get("page_number", 1)
                            }
                            for i, c in enumerate(batch_chunks)
                        ]

                        vector_store.add(batch_ids, batch_embeddings, batch_texts, batch_metadatas)

                        for chunk, emb_id in zip(batch_chunks, batch_ids):
                            update_chunk_embedding(chunk["id"], emb_id)

                        # Release memory between batches
                        del batch_texts, batch_embeddings, batch_ids, batch_metadatas, batch_chunks
                        gc.collect()

                processed += 1

            except Exception as e:
                console.print(f"[red]Error processing {doc_id}: {e}[/red]")
                from src.db.models import update_document_status
                update_document_status(doc_id, "failed")

            progress.advance(task)

    console.print(f"\n[green]Successfully processed {processed} documents[/green]")
    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} documents (no extractable text)[/yellow]")
    console.print(f"Total documents in database: {get_document_count()}")
    console.print(f"Total embeddings in vector store: {vector_store.count()}")


@cli.command()
@click.argument("query")
@click.option("--semantic", "-s", is_flag=True, help="Use semantic search only")
@click.option("--keyword", "-k", is_flag=True, help="Use keyword search only")
@click.option("--limit", "-l", default=10, help="Maximum results")
def search(query: str, semantic: bool, keyword: bool, limit: int):
    """Search documents by query."""
    console.print(Panel(f"Searching: {query}", title="Search"))

    if semantic:
        from src.search import semantic_search
        results = semantic_search(query, limit=limit)
        search_type = "Semantic"
    elif keyword:
        from src.search import keyword_search
        results = keyword_search(query, limit=limit)
        search_type = "Keyword"
    else:
        from src.search import hybrid_search
        results = hybrid_search(query, limit=limit)
        search_type = "Hybrid"

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"[green]{search_type} search returned {len(results)} results:[/green]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Doc ID", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Preview", max_width=60)

    for result in results:
        doc_id = result.get("doc_id", "")
        score = result.get("similarity") or result.get("rrf_score") or result.get("score", 0)
        preview = result.get("snippet") or result.get("chunk_text", "")[:100] or ""

        if isinstance(score, float):
            score_str = f"{score:.4f}"
        else:
            score_str = str(score)

        table.add_row(doc_id, score_str, preview.replace("\n", " ")[:60])

    console.print(table)


@cli.command()
@click.option("--type", "-t", "entity_type", default="PERSON", help="Entity type (PERSON, ORGANIZATION, LOCATION, DATE)")
@click.option("--top", "-n", default=50, help="Number of top entities")
@click.option("--extract", is_flag=True, help="Extract entities from documents")
@click.option("--use-llm", is_flag=True, help="Use LLM for extraction (slower)")
def entities(entity_type: str, top: int, extract: bool, use_llm: bool):
    """View or extract entities from documents."""
    if extract:
        from src.detective import extract_entities_for_document

        console.print(Panel("Extracting entities from documents", title="Entities"))

        docs = get_documents_by_status("completed", limit=1000)
        if not docs:
            console.print("[yellow]No processed documents found[/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(docs))

            total_entities = 0
            for doc in docs:
                count = extract_entities_for_document(
                    doc["doc_id"],
                    use_llm=use_llm
                )
                total_entities += count
                progress.advance(task)

        console.print(f"[green]Extracted {total_entities} entities[/green]")
        return

    # Show top entities
    from src.db.models import get_entities_by_type

    console.print(Panel(f"Top {entity_type} entities", title="Entities"))

    results = get_entities_by_type(entity_type.upper(), limit=top)

    if not results:
        console.print("[yellow]No entities found. Run with --extract first.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Entity", style="cyan")
    table.add_column("Count", justify="right")

    for entity in results:
        table.add_row(entity["entity_value"], str(entity["count"]))

    console.print(table)


@cli.command()
@click.argument("entity")
@click.option("--depth", "-d", default=1, help="Connection depth")
def connections(entity: str, depth: int):
    """Find connections for an entity."""
    from src.detective import find_connections, find_co_occurring_entities

    console.print(Panel(f"Connections for: {entity}", title="Connections"))

    result = find_connections(entity)

    if result["document_count"] == 0:
        console.print(f"[yellow]No documents found mentioning '{entity}'[/yellow]")
        return

    console.print(f"Found in [green]{result['document_count']}[/green] documents")

    if result["documents"]:
        console.print("\n[bold]Documents:[/bold]")
        for doc_id in result["documents"][:10]:
            console.print(f"  - {doc_id}")
        if len(result["documents"]) > 10:
            console.print(f"  ... and {len(result['documents']) - 10} more")

    # Show co-occurring people
    co_people = find_co_occurring_entities(entity, entity_type="PERSON", limit=10)
    if co_people:
        console.print("\n[bold]Frequently appears with (PERSON):[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Person", style="cyan")
        table.add_column("Co-occurrences", justify="right")

        for person in co_people:
            table.add_row(person["entity_value"], str(person["count"]))

        console.print(table)


@cli.command()
@click.option("--from", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--to", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--limit", "-l", default=50, help="Maximum results")
def timeline(from_date: str, to_date: str, limit: int):
    """View documents by date range."""
    from src.detective import timeline_search
    from src.detective.timeline import parse_date

    console.print(Panel("Timeline View", title="Timeline"))

    start = parse_date(from_date) if from_date else None
    end = parse_date(to_date) if to_date else None

    results = timeline_search(start_date=start, end_date=end, limit=limit)

    if not results:
        console.print("[yellow]No dated documents found in the specified range[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Date", style="cyan")
    table.add_column("Doc ID")
    table.add_column("Context", max_width=50)

    for result in results:
        date_str = str(result["parsed_date"]) if result["parsed_date"] else result["date_str"]
        table.add_row(
            date_str,
            result["doc_id"],
            result["context"][:50].replace("\n", " ")
        )

    console.print(table)


@cli.command()
@click.argument("question")
@click.option("--model", "-m", help="LLM model to use")
@click.option("--sources", "-s", is_flag=True, help="Show source documents")
def ask(question: str, model: str, sources: bool):
    """Ask a question using RAG."""
    from src.llm import ask_with_sources

    console.print(Panel(f"Question: {question}", title="Ask"))

    with console.status("Thinking..."):
        result = ask_with_sources(question, model=model)

    console.print("\n[bold]Answer:[/bold]")
    console.print(Markdown(result["answer"]))

    if sources and result["sources"]:
        console.print("\n[bold]Sources:[/bold]")
        for source in result["sources"]:
            relevance = source.get("relevance", 0)
            if isinstance(relevance, float):
                relevance_str = f"{relevance:.3f}"
            else:
                relevance_str = str(relevance)
            console.print(f"  - {source['doc_id']} (relevance: {relevance_str})")


@cli.command()
@click.argument("doc_id")
def show(doc_id: str):
    """Show details for a specific document."""
    doc = get_document(doc_id)

    if not doc:
        console.print(f"[red]Document not found: {doc_id}[/red]")
        return

    console.print(Panel(f"Document: {doc_id}", title="Document Details"))

    console.print(f"[bold]File:[/bold] {doc.get('file_path', 'N/A')}")
    console.print(f"[bold]Volume:[/bold] {doc.get('volume', 'N/A')}")
    console.print(f"[bold]Pages:[/bold] {doc.get('page_count', 'N/A')}")
    console.print(f"[bold]Status:[/bold] {doc.get('ocr_status', 'N/A')}")

    if doc.get("extracted_text"):
        text = doc["extracted_text"]
        preview = text[:1000] + "..." if len(text) > 1000 else text
        console.print(f"\n[bold]Text Preview:[/bold]")
        console.print(preview)


@cli.command()
def status():
    """Show indexing status."""
    console.print(Panel("System Status", title="Status"))

    total = get_document_count()
    pending = len(get_documents_by_status("pending", limit=10000))
    completed = len(get_documents_by_status("completed", limit=10000))
    failed = len(get_documents_by_status("failed", limit=10000))

    console.print(f"[bold]Total documents:[/bold] {total}")
    console.print(f"[bold]Completed:[/bold] [green]{completed}[/green]")
    console.print(f"[bold]Pending:[/bold] [yellow]{pending}[/yellow]")
    console.print(f"[bold]Failed:[/bold] [red]{failed}[/red]")

    # Show counts by file type
    type_counts = count_documents_by_file_type()
    if type_counts:
        console.print("\n[bold]By file type:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Type", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Completed", justify="right", style="green")
        table.add_column("Pending", justify="right", style="yellow")
        table.add_column("Failed", justify="right", style="red")

        for row in type_counts:
            table.add_row(
                row["file_type"],
                str(row["total"]),
                str(row["completed"]),
                str(row["pending"]),
                str(row["failed"]),
            )
        console.print(table)

    try:
        from src.indexing import VectorStore
        store = VectorStore()
        console.print(f"[bold]Embeddings:[/bold] {store.count()}")
    except Exception:
        console.print("[bold]Embeddings:[/bold] N/A")

    try:
        from src.db.models import get_image_count
        console.print(f"[bold]Images:[/bold] {get_image_count()}")
    except Exception:
        console.print("[bold]Images:[/bold] N/A")


@cli.command("extract-images")
@click.option("--limit", "-l", default=100, help="Maximum documents to process")
@click.option("--describe", "-d", is_flag=True, help="Generate AI descriptions for images")
@click.option("--model", "-m", help="Vision model for descriptions")
@click.option("--output-dir", "-o", type=click.Path(), help="Save images to directory")
def extract_images(limit: int, describe: bool, model: str, output_dir: str):
    """Extract images from PDFs and optionally describe them."""
    from src.extraction import extract_images_from_pdf, describe_image_for_search
    from src.db.models import create_image, get_images_for_document
    from src.config import DB_DIR

    console.print(Panel("Extracting images from documents", title="Images"))

    # Default output directory
    if output_dir:
        img_output = Path(output_dir)
    else:
        img_output = DB_DIR / "images"
    img_output.mkdir(parents=True, exist_ok=True)

    # Get completed documents
    docs = get_documents_by_status("completed", limit=limit)
    if not docs:
        console.print("[yellow]No processed documents found[/yellow]")
        return

    console.print(f"Processing {len(docs)} documents...")

    total_images = 0
    total_described = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting images...", total=len(docs))

        for doc in docs:
            doc_id = doc["doc_id"]
            file_path = doc["file_path"]

            progress.update(task, description=f"Processing {doc_id}...")

            # Check if images already extracted for this doc
            existing_images = get_images_for_document(doc["id"])
            if existing_images:
                progress.advance(task)
                continue

            try:
                for page_num, img_idx, image, meta in extract_images_from_pdf(file_path):
                    # Save image to disk
                    img_filename = f"{doc_id}_p{page_num}_i{img_idx}.png"
                    img_path = img_output / img_filename
                    image.save(str(img_path))

                    # Generate description if requested
                    description = ""
                    if describe:
                        try:
                            description = describe_image_for_search(image, model=model)
                            total_described += 1
                        except Exception as e:
                            console.print(f"[yellow]Failed to describe {img_filename}: {e}[/yellow]")

                    # Store in database
                    create_image(
                        document_id=doc["id"],
                        page_number=page_num,
                        image_index=img_idx,
                        width=meta["width"],
                        height=meta["height"],
                        file_path=str(img_path),
                        description=description
                    )
                    total_images += 1

            except Exception as e:
                console.print(f"[red]Error extracting from {doc_id}: {e}[/red]")

            progress.advance(task)

    console.print(f"\n[green]Extracted {total_images} images[/green]")
    if describe:
        console.print(f"[green]Generated {total_described} descriptions[/green]")
    console.print(f"Images saved to: {img_output}")


@cli.command("describe-images")
@click.option("--limit", "-l", default=100, help="Maximum images to describe")
@click.option("--model", "-m", help="Vision model to use")
def describe_images(limit: int, model: str):
    """Generate descriptions for images without them."""
    from src.extraction import describe_image_for_search
    from src.db.models import get_images_without_descriptions, update_image_description

    console.print(Panel("Generating image descriptions", title="Describe"))

    images = get_images_without_descriptions(limit=limit)
    if not images:
        console.print("[green]All images already have descriptions[/green]")
        return

    console.print(f"Found {len(images)} images without descriptions")

    described = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Describing images...", total=len(images))

        for img in images:
            img_path = img["file_path"]
            progress.update(task, description=f"Describing {Path(img_path).name}...")

            try:
                description = describe_image_for_search(img_path, model=model)
                update_image_description(img["id"], description)
                described += 1
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

            progress.advance(task)

    console.print(f"\n[green]Described {described} images[/green]")


@cli.command("search-images")
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Maximum results")
def search_images_cmd(query: str, limit: int):
    """Search images by description."""
    from src.db.models import search_images

    console.print(Panel(f"Searching images: {query}", title="Image Search"))

    results = search_images(query, limit=limit)

    if not results:
        console.print("[yellow]No images found[/yellow]")
        return

    console.print(f"[green]Found {len(results)} images:[/green]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Doc ID", style="cyan")
    table.add_column("Page")
    table.add_column("Size")
    table.add_column("Description", max_width=50)

    for img in results:
        desc = img.get("snippet") or img.get("description", "")[:50]
        table.add_row(
            img["doc_id"],
            str(img["page_number"]),
            f"{img['width']}x{img['height']}",
            desc.replace("\n", " ")
        )

    console.print(table)


@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the web interface."""
    import uvicorn

    console.print(Panel(f"Starting web server at http://{host}:{port}", title="Web Server"))
    console.print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
