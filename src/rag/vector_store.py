"""Chroma vector database for storing and querying chunk embeddings."""

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import chromadb
import numpy as np
import structlog
from chromadb.api.models.Collection import Collection

from src.utils.exceptions import VectorStoreError


class ChunkData(TypedDict):
    """Data structure for chunk storage.

    Attributes:
        id: Unique chunk identifier (wiki_page_id_chunk_index format)
        wiki_page_id: Wiki page ID for deduplication/updates
        article_title: Title of the source article
        section_path: Hierarchical section path, "Infobox" for infobox chunks
        chunk_text: The actual text content of the chunk
        chunk_index: Zero-based index of chunk within article
        metadata: Flexible metadata dict (article_last_updated, links, etc.)
    """

    id: str
    wiki_page_id: str
    article_title: str
    section_path: str
    chunk_text: str
    chunk_index: int
    metadata: dict[str, Any]


class ChunkMetadata(TypedDict, total=False):
    """Metadata schema for chunk storage in Chroma.

    Attributes:
        wiki_page_id: Wiki page ID for deduplication/updates (required)
        article_title: Title of the source article (required)
        section_path: Hierarchical section path, "Infobox" for infobox chunks (required)
        chunk_index: Zero-based index of chunk within article (required)
        article_last_updated: ISO timestamp of last article update (required)
        links: Internal wiki links found in this chunk (optional)
    """

    wiki_page_id: str
    article_title: str
    section_path: str
    chunk_index: int
    article_last_updated: str
    links: list[str]


class ChromaVectorStore:
    """Vector database for storing and querying chunk embeddings using Chroma.

    This class provides a high-level interface to Chroma for storing chunk data
    and performing similarity searches with metadata filtering.

    Attributes:
        client: Chroma PersistentClient instance
        collection: Chroma collection for storing chunks
        storage_path: Path to Chroma database storage
        collection_name: Name of the Chroma collection

    Example:
        >>> store = ChromaVectorStore()
        >>> embeddings = [[0.1, 0.2, ...], ...]
        >>> store.add_chunks(chunks, embeddings)
        >>> results = store.query(
        ...     query_embedding=[0.1, 0.2, ...],
        ...     n_results=10,
        ...     filters={"section_path": "Infobox"}
        ... )
    """

    # Configuration constants
    DEFAULT_STORAGE_PATH = "data/chroma-db/"
    DEFAULT_COLLECTION_NAME = "wh40k-lore"
    DEFAULT_DISTANCE_METRIC = "cosine"
    BATCH_SIZE = 1000  # Chroma recommended batch size

    def __init__(
        self,
        storage_path: str = DEFAULT_STORAGE_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize ChromaVectorStore.

        Args:
            storage_path: Path to Chroma database storage (default: "data/chroma-db/")
            collection_name: Name of the Chroma collection (default: "wh40k-lore")

        Raises:
            VectorStoreError: If initialization fails
        """
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.logger = structlog.get_logger(__name__)

        try:
            # Ensure storage directory exists
            Path(storage_path).mkdir(parents=True, exist_ok=True)

            # Initialize persistent client
            self.client = chromadb.PersistentClient(path=storage_path)

            # Get or create collection with cosine similarity
            self.collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.DEFAULT_DISTANCE_METRIC},
            )

            self.logger.info(
                "vector_store_initialized",
                storage_path=storage_path,
                collection_name=collection_name,
                distance_metric=self.DEFAULT_DISTANCE_METRIC,
            )

        except Exception as e:
            self.logger.error(
                "vector_store_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to initialize Chroma vector store: {e}") from e

    def add_chunks(
        self,
        chunks: list[ChunkData],
        embeddings: list[np.ndarray],
    ) -> None:
        """Add chunks with embeddings to the vector store.

        Chunks are inserted in batches of 1000 for optimal performance.

        Args:
            chunks: List of ChunkData dicts to store
            embeddings: List of embeddings (1536-dim numpy arrays)

        Raises:
            VectorStoreError: If chunks and embeddings lengths don't match
            VectorStoreError: If insertion fails
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunks and embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
            )

        if not chunks:
            self.logger.warning("add_chunks_called_with_empty_list")
            return

        self.logger.info("adding_chunks_to_vector_store", total_chunks=len(chunks))

        try:
            # Process in batches
            total_batches = (len(chunks) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            for batch_idx in range(0, len(chunks), self.BATCH_SIZE):
                batch_end = min(batch_idx + self.BATCH_SIZE, len(chunks))
                batch_chunks = chunks[batch_idx:batch_end]
                batch_embeddings = embeddings[batch_idx:batch_end]

                # Prepare data for Chroma
                ids = [chunk["id"] for chunk in batch_chunks]
                embeddings_list = [emb.tolist() for emb in batch_embeddings]
                metadatas = [self._chunk_to_metadata(chunk) for chunk in batch_chunks]
                documents = [chunk["chunk_text"] for chunk in batch_chunks]

                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,  # type: ignore[arg-type]
                    documents=documents,
                )

                batch_number = batch_idx // self.BATCH_SIZE + 1
                self.logger.info(
                    "batch_added_to_vector_store",
                    batch_number=batch_number,
                    total_batches=total_batches,
                    batch_size=len(batch_chunks),
                )

            self.logger.info("chunks_added_successfully", total_chunks=len(chunks))

        except Exception as e:
            self.logger.error(
                "chunk_insertion_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to add chunks to vector store: {e}") from e

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ChunkData, float]]:
        """Query vector store for similar chunks.

        Args:
            query_embedding: Query embedding (1536-dim numpy array)
            n_results: Number of results to return (default: 10)
            filters: Metadata filters (default: None)
                Examples:
                - {"wiki_page_id": "123"}
                - {"article_title": "Ultramarines"}
                - {"section_path": "Infobox"}

        Returns:
            List of tuples (ChunkData, distance_score) sorted by similarity

        Raises:
            VectorStoreError: If query fails

        Example:
            >>> results = store.query(
            ...     query_embedding=embedding,
            ...     n_results=10,
            ...     filters={"section_path": "Infobox"}
            ... )
            >>> for chunk, score in results:
            ...     print(f"{chunk['article_title']}: {score}")
        """
        try:
            self.logger.info(
                "querying_vector_store",
                n_results=n_results,
                filters=filters,
            )

            # Convert compound filters to Chroma $and format
            chroma_filters = self._convert_filters_to_chroma_format(filters)

            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=chroma_filters,
            )

            # Convert results to ChunkData dicts
            chunks_with_scores: list[tuple[ChunkData, float]] = []

            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    # Reconstruct ChunkData from metadata
                    chunk = self._metadata_to_chunk(
                        chunk_id=chunk_id,
                        metadata=cast(dict[str, Any], metadata),
                        document=document,
                    )

                    chunks_with_scores.append((chunk, distance))

            self.logger.info(
                "query_completed",
                results_found=len(chunks_with_scores),
            )

            return chunks_with_scores

        except Exception as e:
            self.logger.error(
                "query_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to query vector store: {e}") from e

    def count(self) -> int:
        """Get total number of chunks in the collection.

        Returns:
            Total number of chunks stored

        Raises:
            VectorStoreError: If count operation fails
        """
        try:
            count = self.collection.count()
            self.logger.info("collection_count", total_chunks=count)
            return count
        except Exception as e:
            self.logger.error(
                "count_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to get collection count: {e}") from e

    def delete_collection(self) -> None:
        """Delete the collection (for testing/reset purposes).

        Warning:
            This permanently deletes all data in the collection.

        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.warning(
                "collection_deleted",
                collection_name=self.collection_name,
            )
        except Exception as e:
            self.logger.error(
                "collection_deletion_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to delete collection: {e}") from e

    def delete_by_wiki_page_id(self, wiki_page_id: str) -> int:
        """Delete all chunks belonging to a specific wiki page.

        Used for re-ingestion to remove old chunks before adding updated ones.

        Args:
            wiki_page_id: Wiki page ID to delete chunks for

        Returns:
            Number of chunks deleted

        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            # First, count how many chunks exist for this wiki_page_id
            results = self.collection.get(
                where={"wiki_page_id": wiki_page_id},
                include=[],  # Don't need embeddings or documents, just IDs
            )

            chunk_count = len(results["ids"]) if results["ids"] else 0

            if chunk_count == 0:
                self.logger.info(
                    "no_chunks_to_delete",
                    wiki_page_id=wiki_page_id,
                )
                return 0

            # Delete all chunks for this wiki_page_id
            self.collection.delete(where={"wiki_page_id": wiki_page_id})

            self.logger.info(
                "chunks_deleted_for_wiki_page",
                wiki_page_id=wiki_page_id,
                chunks_deleted=chunk_count,
            )

            return chunk_count

        except Exception as e:
            self.logger.error(
                "delete_by_wiki_page_id_failed",
                wiki_page_id=wiki_page_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(
                f"Failed to delete chunks for wiki_page_id {wiki_page_id}: {e}"
            ) from e

    def get_by_id(self, chunk_id: str) -> ChunkData | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            ChunkData if found, None otherwise

        Raises:
            VectorStoreError: If retrieval fails
        """
        try:
            results = self.collection.get(ids=[chunk_id])

            if not results["ids"]:
                self.logger.info("chunk_not_found", chunk_id=chunk_id)
                return None

            metadata = results["metadatas"][0] if results["metadatas"] else {}
            document = results["documents"][0] if results["documents"] else ""

            chunk = self._metadata_to_chunk(
                chunk_id=chunk_id,
                metadata=cast(dict[str, Any], metadata),
                document=document,
            )

            self.logger.info("chunk_retrieved", chunk_id=chunk_id)
            return chunk

        except Exception as e:
            self.logger.error(
                "get_by_id_failed",
                chunk_id=chunk_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to get chunk by ID: {e}") from e

    def get_article_last_updated(self, wiki_page_id: str) -> str | None:
        """Get the stored last_updated timestamp for an article.

        Used for change detection during re-ingestion to skip unchanged articles.

        Args:
            wiki_page_id: Wiki page ID to check

        Returns:
            ISO timestamp string if article exists, None if not found

        Raises:
            VectorStoreError: If query fails
        """
        try:
            # Get first chunk for this article (all chunks have same last_updated)
            results = self.collection.get(
                where={"wiki_page_id": wiki_page_id},
                limit=1,
                include=["metadatas"],
            )

            if not results["ids"]:
                self.logger.debug(
                    "article_not_found_in_vector_store",
                    wiki_page_id=wiki_page_id,
                )
                return None

            metadata = results["metadatas"][0] if results["metadatas"] else {}
            last_updated = metadata.get("article_last_updated")

            self.logger.debug(
                "article_last_updated_retrieved",
                wiki_page_id=wiki_page_id,
                last_updated=last_updated,
            )
            return str(last_updated) if last_updated is not None else None

        except Exception as e:
            self.logger.error(
                "get_article_last_updated_failed",
                wiki_page_id=wiki_page_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise VectorStoreError(
                f"Failed to get article last_updated for {wiki_page_id}: {e}"
            ) from e

    def _convert_filters_to_chroma_format(
        self, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Convert simple filters to Chroma filter format.

        Chroma requires compound filters (multiple keys) to use $and operator.
        This method converts simple dict filters to the proper format.

        Args:
            filters: Simple filter dict (e.g., {"faction": "X", "spoiler_flag": False})

        Returns:
            Chroma-formatted filter dict or None

        Examples:
            >>> _convert_filters_to_chroma_format({"faction": "Space Marines"})
            {"faction": "Space Marines"}  # Single filter unchanged

            >>> _convert_filters_to_chroma_format({
            ...     "faction": "Space Marines",
            ...     "spoiler_flag": False
            ... })
            {"$and": [{"faction": "Space Marines"}, {"spoiler_flag": False}]}
        """
        if not filters:
            return None

        # Single filter - return as-is
        if len(filters) == 1:
            return filters

        # Multiple filters - convert to $and format
        filter_list = [{key: value} for key, value in filters.items()]
        return {"$and": filter_list}

    def _chunk_to_metadata(self, chunk: ChunkData) -> dict[str, Any]:
        """Convert ChunkData to Chroma metadata format.

        Args:
            chunk: ChunkData dict

        Returns:
            Metadata dictionary for Chroma (no None values allowed)
        """
        metadata: dict[str, Any] = {
            "wiki_page_id": chunk["wiki_page_id"],
            "article_title": chunk["article_title"],
            "section_path": chunk["section_path"],
            "chunk_index": chunk["chunk_index"],
        }

        # Add article_last_updated for change detection (required for re-ingestion)
        article_last_updated = chunk["metadata"].get("article_last_updated")
        if article_last_updated is not None:
            metadata["article_last_updated"] = article_last_updated

        # Add links if present (stored as JSON string for Chroma compatibility)
        links = chunk["metadata"].get("links")
        if links is not None and links:
            # Chroma doesn't support list types in metadata, store as JSON string
            metadata["links"] = json.dumps(links)

        return metadata

    def _metadata_to_chunk(
        self,
        chunk_id: str,
        metadata: dict[str, Any],
        document: str,
    ) -> ChunkData:
        """Convert Chroma metadata back to ChunkData.

        Args:
            chunk_id: Unique chunk identifier
            metadata: Chroma metadata dictionary
            document: Chunk text content

        Returns:
            ChunkData dict
        """
        # Build metadata dict from Chroma metadata
        chunk_metadata: dict[str, Any] = {}

        # Add article_last_updated if present (for change detection)
        if "article_last_updated" in metadata:
            chunk_metadata["article_last_updated"] = metadata["article_last_updated"]

        # Parse links from JSON string if present
        if "links" in metadata:
            try:
                chunk_metadata["links"] = json.loads(metadata["links"])
            except (json.JSONDecodeError, TypeError):
                chunk_metadata["links"] = []

        # Create ChunkData dict
        chunk: ChunkData = {
            "id": chunk_id,
            "wiki_page_id": metadata.get("wiki_page_id", ""),
            "article_title": metadata.get("article_title", ""),
            "section_path": metadata.get("section_path", ""),
            "chunk_text": document,
            "chunk_index": metadata.get("chunk_index", 0),
            "metadata": chunk_metadata,
        }

        return chunk
