"""Integration tests for embedding generation with real OpenAI API."""

import os
import time

import numpy as np
import pytest
from dotenv import load_dotenv

from src.ingestion.embedding_generator import EmbeddingGenerator

# Load .env file for integration tests
load_dotenv()


@pytest.mark.integration
class TestEmbeddingGeneratorIntegration:
    """Integration tests using real OpenAI API."""

    @pytest.fixture
    def generator(self):
        """Create EmbeddingGenerator instance."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return EmbeddingGenerator(api_key=api_key)

    def test_generate_embeddings_real_api(self, generator):
        """Test embedding generation with real API (small batch)."""
        # Use small batch of 10 chunks to minimize cost
        chunks = [
            "The Emperor of Mankind sits upon the Golden Throne.",
            "Roboute Guilliman is the Primarch of the Ultramarines.",
            "The Horus Heresy was a galaxy-spanning civil war.",
            "Chaos Space Marines serve the dark gods.",
            "The Imperial Guard defends humanity across millions of worlds.",
            "Orks live for war and combat.",
            "The Eldar are an ancient and dying race.",
            "Necrons are ancient robotic warriors.",
            "Tau seek to expand the Greater Good.",
            "Tyranids are an extragalactic hive mind.",
        ]

        # Generate embeddings
        embeddings = generator.generate_embeddings(chunks)

        # Verify results
        assert len(embeddings) == 10, "Should return 10 embeddings"
        assert all(e is not None for e in embeddings), "All embeddings should be generated"

        # Check embedding properties
        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, np.ndarray), f"Embedding {i} should be numpy array"
            assert embedding.shape == (1536,), f"Embedding {i} should have 1536 dimensions"
            assert embedding.dtype == np.float32, f"Embedding {i} should be float32"

            # Check L2 normalization (OpenAI embeddings are normalized)
            norm = np.linalg.norm(embedding)
            assert 0.99 <= norm <= 1.01, (
                f"Embedding {i} should be approximately normalized (norm={norm})"
            )

        # Verify cost tracking
        assert generator.total_tokens > 0, "Should track tokens"
        assert generator.total_cost > 0, "Should track cost"

        # Log cost for visibility
        summary = generator.get_cost_summary()
        print("\nIntegration test cost summary:")
        print(f"  Total tokens: {summary['total_tokens']}")
        print(f"  Total cost: ${summary['total_cost_usd']:.6f}")
        print(f"  Cost per 1K tokens: ${summary['cost_per_1k_tokens']:.6f}")

        # Cost should be minimal (10 small chunks)
        assert summary["total_cost_usd"] < 0.01, "Cost should be less than 1 cent"

    def test_batch_processing_real_api(self, generator):
        """Test batch processing with real API."""
        # Create 15 chunks to test batching (but stay small for cost)
        chunks = [f"Test chunk number {i} about Warhammer 40K lore." for i in range(15)]

        embeddings = generator.generate_embeddings(chunks)

        # Verify all embeddings generated
        assert len(embeddings) == 15
        assert all(e is not None for e in embeddings)

        # Verify unique embeddings (each chunk should get different embedding)
        # Check first and last are different
        similarity = np.dot(embeddings[0], embeddings[-1])
        assert similarity < 0.99, "Different chunks should have different embeddings"

    def test_empty_chunk_handling(self, generator):
        """Test handling of empty or invalid chunks."""
        chunks = [
            "Valid chunk with content.",
            "",  # Empty
            "Another valid chunk.",
            "   ",  # Whitespace only
            "Final valid chunk.",
        ]

        embeddings = generator.generate_embeddings(chunks)

        # Should return None for invalid chunks
        assert len(embeddings) == 5
        assert embeddings[0] is not None, "First chunk should have embedding"
        assert embeddings[1] is None, "Empty chunk should be None"
        assert embeddings[2] is not None, "Third chunk should have embedding"
        assert embeddings[3] is None, "Whitespace chunk should be None"
        assert embeddings[4] is not None, "Last chunk should have embedding"

    def test_cost_calculation_accuracy(self, generator):
        """Test that cost calculation matches expected rates."""
        # Single small chunk
        chunks = ["Test"]

        generator.generate_embeddings(chunks)

        # Get cost summary
        summary = generator.get_cost_summary()

        # Verify cost calculation formula
        expected_cost = summary["total_tokens"] * (0.02 / 1_000_000)
        assert abs(summary["total_cost_usd"] - expected_cost) < 0.000001, (
            "Cost calculation should match formula"
        )

        # Cost per 1K tokens should be $0.00002
        assert summary["cost_per_1k_tokens"] == 0.00002

    @pytest.mark.skip(reason="Slow test - only run manually to verify rate limiting")
    def test_rate_limiting_real(self, generator):
        """Test rate limiting with real API (SLOW - skip by default)."""
        # This test would make many requests to verify rate limiting
        # Skip by default to avoid slow test runs and API costs
        chunks = [f"Chunk {i}" for i in range(10)]

        start = time.time()
        generator.generate_embeddings(chunks)
        elapsed = time.time() - start

        # With rate limiting, should not make requests too fast
        print(f"\nGenerated 10 embeddings in {elapsed:.2f} seconds")
        assert generator.total_tokens > 0
