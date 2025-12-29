"""Chroma vector database for storing and querying chunk embeddings."""

from pathlib import Path
from typing import Any, TypedDict, cast

import chromadb
import numpy as np
import structlog
from chromadb.api.models.Collection import Collection

from src.models.wiki_chunk import WikiChunk
from src.utils.exceptions import VectorStoreError


class ChunkMetadata(TypedDict, total=False):
    """Metadata schema for chunk storage in Chroma.

    Attributes:
        article_title: Title of the source article (required)
        section_path: Hierarchical section path (required)
        chunk_index: Zero-based index of chunk within article (required)
        faction: Faction tag for filtering (optional)
        era: Era tag for filtering (optional)
        spoiler_flag: Whether chunk contains spoilers (required)
        content_type: Type of content (e.g., "lore", "rules") (required)
    """

    article_title: str
    section_path: str
    chunk_index: int
    faction: str  # Optional
    era: str  # Optional
    spoiler_flag: bool
    content_type: str


class ChromaVectorStore:
    """Vector database for storing and querying chunk embeddings using Chroma.

    This class provides a high-level interface to Chroma for storing WikiChunk
    embeddings and performing similarity searches with metadata filtering.

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
        ...     filters={"faction": "Space Marines", "spoiler_flag": False}
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
        chunks: list[WikiChunk],
        embeddings: list[np.ndarray],
    ) -> None:
        """Add chunks with embeddings to the vector store.

        Chunks are inserted in batches of 1000 for optimal performance.

        Args:
            chunks: List of WikiChunk objects to store
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
                ids = [chunk.id for chunk in batch_chunks]
                embeddings_list = [emb.tolist() for emb in batch_embeddings]
                metadatas = [self._chunk_to_metadata(chunk) for chunk in batch_chunks]
                documents = [chunk.chunk_text for chunk in batch_chunks]

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
    ) -> list[tuple[WikiChunk, float]]:
        """Query vector store for similar chunks.

        Args:
            query_embedding: Query embedding (1536-dim numpy array)
            n_results: Number of results to return (default: 10)
            filters: Metadata filters (default: None)
                Examples:
                - {"faction": "Space Marines"}
                - {"era": "Horus Heresy"}
                - {"spoiler_flag": False}
                - {"faction": "Space Marines", "spoiler_flag": False}

        Returns:
            List of tuples (WikiChunk, distance_score) sorted by similarity

        Raises:
            VectorStoreError: If query fails

        Example:
            >>> results = store.query(
            ...     query_embedding=embedding,
            ...     n_results=10,
            ...     filters={"faction": "Space Marines", "spoiler_flag": False}
            ... )
            >>> for chunk, score in results:
            ...     print(f"{chunk.article_title}: {score}")
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

            # Convert results to WikiChunk objects
            chunks_with_scores: list[tuple[WikiChunk, float]] = []

            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    # Reconstruct WikiChunk from metadata
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

    def get_by_id(self, chunk_id: str) -> WikiChunk | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            WikiChunk if found, None otherwise

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

    def _chunk_to_metadata(self, chunk: WikiChunk) -> dict[str, Any]:
        """Convert WikiChunk to Chroma metadata format.

        Args:
            chunk: WikiChunk object

        Returns:
            Metadata dictionary for Chroma
        """
        metadata: dict[str, Any] = {
            "article_title": chunk.article_title,
            "section_path": chunk.section_path,
            "chunk_index": chunk.chunk_index,
            "spoiler_flag": chunk.metadata_json.get("spoiler_flag", False),
            "content_type": chunk.metadata_json.get("content_type", "lore"),
        }

        # Add optional fields if present
        if "faction" in chunk.metadata_json:
            metadata["faction"] = chunk.metadata_json["faction"]

        if "era" in chunk.metadata_json:
            metadata["era"] = chunk.metadata_json["era"]

        return metadata

    def _metadata_to_chunk(
        self,
        chunk_id: str,
        metadata: dict[str, Any],
        document: str,
    ) -> WikiChunk:
        """Convert Chroma metadata back to WikiChunk.

        Args:
            chunk_id: Unique chunk identifier
            metadata: Chroma metadata dictionary
            document: Chunk text content

        Returns:
            WikiChunk object
        """
        # Build metadata_json from Chroma metadata
        metadata_json: dict[str, Any] = {
            "spoiler_flag": metadata.get("spoiler_flag", False),
            "content_type": metadata.get("content_type", "lore"),
        }

        # Add optional fields if present
        if "faction" in metadata:
            metadata_json["faction"] = metadata["faction"]

        if "era" in metadata:
            metadata_json["era"] = metadata["era"]

        # Create WikiChunk
        chunk = WikiChunk(
            id=chunk_id,
            wiki_page_id="",  # Not stored in Chroma
            article_title=metadata.get("article_title", ""),
            section_path=metadata.get("section_path", ""),
            chunk_text=document,
            chunk_index=metadata.get("chunk_index", 0),
            metadata_json=metadata_json,
        )

        return chunk
