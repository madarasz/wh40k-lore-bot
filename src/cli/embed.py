"""CLI command for generating embeddings."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import structlog
from dotenv import load_dotenv
from tqdm import tqdm

from src.ingestion.embedding_generator import EmbeddingGenerator
from src.utils.chunk_id import generate_chunk_id

# Load environment variables from .env file
load_dotenv()

logger = structlog.get_logger(__name__)


@click.command()
@click.argument("chunks_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/embeddings.json",
    help="Output file path for embeddings JSON (default: data/embeddings.json)",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Number of chunks to embed per batch (default: 100)",
)
def embed(  # noqa: PLR0915
    chunks_file: Path,
    output: Path,
    batch_size: int,
) -> None:
    """Generate embeddings for chunks.

    Reads chunks JSON file (from the chunk command) and generates embeddings
    using the OpenAI API. Outputs embeddings JSON for the store step.

    CHUNKS_FILE: Path to the chunks JSON file (output of 'chunk' command)

    Examples:

        \b
        # Generate embeddings for chunks
        poetry run embed data/chunks.json

        \b
        # Custom output and batch size
        poetry run embed data/chunks.json --output data/my-embeddings.json --batch-size 50
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Embedding Generation")
    click.echo("=" * 80)
    click.echo()

    click.echo(f"  Chunks File: {chunks_file}")
    click.echo(f"  Output File: {output}")
    click.echo(f"  Batch Size: {batch_size}")
    click.echo()

    # Load chunks
    click.echo("Loading chunks...")
    try:
        with chunks_file.open("r", encoding="utf-8") as f:
            chunks_data = json.load(f)
    except Exception as e:
        click.echo(f"  Failed to load chunks: {e}", err=True)
        raise click.Abort() from e

    chunks = chunks_data.get("chunks", [])
    if not chunks:
        click.echo("  No chunks found in file", err=True)
        raise click.Abort()

    click.echo(f"  Loaded {len(chunks)} chunks")
    click.echo()

    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()

    # Process in batches
    embeddings_data: list[dict[str, Any]] = []
    total_chunks = len(chunks)
    processed = 0

    click.echo("Generating embeddings...")

    try:
        for i in tqdm(range(0, total_chunks, batch_size), desc="Batches", unit="batch"):
            batch = chunks[i : i + batch_size]
            batch_texts = [chunk["chunk_text"] for chunk in batch]

            # Generate embeddings
            batch_embeddings = embedding_generator.generate_embeddings(batch_texts)

            # Pair embeddings with chunk data
            for chunk, embedding in zip(batch, batch_embeddings, strict=False):
                if embedding is not None:
                    # Generate deterministic chunk ID
                    chunk_id = generate_chunk_id(chunk["wiki_page_id"], chunk["chunk_index"])

                    embedding_entry = {
                        "chunk_id": chunk_id,
                        "wiki_page_id": chunk["wiki_page_id"],
                        "article_title": chunk["article_title"],
                        "section_path": chunk["section_path"],
                        "chunk_index": chunk["chunk_index"],
                        "chunk_text": chunk["chunk_text"],
                        "metadata": chunk.get("metadata", {}),
                        "embedding": embedding.tolist(),
                    }
                    embeddings_data.append(embedding_entry)

            processed += len(batch)

    except KeyboardInterrupt:
        click.echo()
        click.echo("  Interrupted by user", err=True)
        raise click.Abort() from None
    except Exception as e:
        click.echo()
        click.echo(f"  Embedding generation failed: {e}", err=True)
        logger.error("embedding_failed", error=str(e), exc_info=True)
        raise click.Abort() from e

    # Get cost summary
    cost_summary = embedding_generator.get_cost_summary()

    # Create output structure
    output_data = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source_file": str(chunks_file),
        "model": EmbeddingGenerator.MODEL_NAME,
        "dimensions": EmbeddingGenerator.EMBEDDING_DIMENSIONS,
        "total_embeddings": len(embeddings_data),
        "tokens_used": cost_summary["total_tokens"],
        "cost_usd": round(cost_summary["total_cost_usd"], 4),
        "embeddings": embeddings_data,
    }

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    click.echo()
    click.echo("Writing output file...")

    with output.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    click.echo()
    click.echo("=" * 80)
    click.echo("Embedding Generation Complete!")
    click.echo("=" * 80)
    click.echo(f"  Chunks Processed: {processed}")
    click.echo(f"  Embeddings Generated: {len(embeddings_data)}")
    click.echo(f"  Tokens Used: {cost_summary['total_tokens']:,}")
    click.echo(f"  Cost: ${cost_summary['total_cost_usd']:.4f} USD")
    click.echo(f"  Output File: {output}")
    click.echo()


if __name__ == "__main__":
    embed()
