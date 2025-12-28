"""Unit tests for EmbeddingGenerator."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from openai import AuthenticationError, OpenAIError, RateLimitError

from src.ingestion.embedding_generator import EmbeddingGenerator, RateLimiter
from src.utils.exceptions import ConfigurationError, EmbeddingGenerationError


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_rpm=100)
        assert limiter.max_rpm == 100
        assert len(limiter.requests) == 0

    def test_wait_if_needed_under_limit(self):
        """Test no wait when under rate limit."""
        limiter = RateLimiter(max_rpm=10)

        # Add 5 requests
        for _ in range(5):
            limiter.wait_if_needed()

        # Should have 5 requests logged
        assert len(limiter.requests) == 5

    @patch("time.sleep")
    def test_wait_if_needed_at_limit(self, mock_sleep):
        """Test wait when at rate limit."""
        limiter = RateLimiter(max_rpm=5)

        # Fill up the limit with current timestamps
        current_time = time.time()
        for _ in range(5):
            limiter.requests.append(current_time)

        # Next request should trigger sleep
        limiter.wait_if_needed()

        # Should have called sleep (would wait ~60 seconds for oldest to expire)
        assert mock_sleep.call_count == 1
        # Sleep time should be close to 60 seconds (time until oldest request expires)
        sleep_time = mock_sleep.call_args[0][0]
        assert 55 <= sleep_time <= 61  # Allow some timing variance

    def test_cleanup_old_requests(self):
        """Test that old requests are removed."""
        limiter = RateLimiter(max_rpm=100)

        # Add old requests (more than 60 seconds ago)
        old_time = time.time() - 61
        for _ in range(10):
            limiter.requests.append(old_time)

        # Add new request
        limiter.wait_if_needed()

        # Old requests should be cleaned up
        assert len(limiter.requests) == 1


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        generator = EmbeddingGenerator(api_key="test-key")
        assert generator.api_key == "test-key"
        assert generator.client is not None
        assert generator.total_tokens == 0
        assert generator.total_cost == 0.0

    def test_init_without_api_key_raises_error(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                EmbeddingGenerator()
            assert "OPENAI_API_KEY not set" in str(exc_info.value)

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            generator = EmbeddingGenerator()
            assert generator.api_key == "env-key"

    def test_generate_embeddings_empty_list(self):
        """Test handling of empty chunks list."""
        generator = EmbeddingGenerator(api_key="test-key")
        result = generator.generate_embeddings([])
        assert result == []

    def test_generate_embeddings_all_invalid(self):
        """Test handling when all chunks are invalid."""
        generator = EmbeddingGenerator(api_key="test-key")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            generator.generate_embeddings(["", "  ", None])

        assert "All chunks are empty or invalid" in str(exc_info.value)

    @patch("src.ingestion.embedding_generator.OpenAI")
    def test_generate_embeddings_success(self, mock_openai_class):
        """Test successful embedding generation."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_embedding_1 = Mock()
        mock_embedding_1.embedding = [0.1] * 1536
        mock_embedding_2 = Mock()
        mock_embedding_2.embedding = [0.2] * 1536

        mock_response = Mock()
        mock_response.data = [mock_embedding_1, mock_embedding_2]
        mock_response.usage.total_tokens = 100

        mock_client.embeddings.create.return_value = mock_response

        # Test
        generator = EmbeddingGenerator(api_key="test-key")
        chunks = ["chunk 1", "chunk 2"]
        embeddings = generator.generate_embeddings(chunks)

        # Verify
        assert len(embeddings) == 2
        assert embeddings[0] is not None
        assert embeddings[1] is not None
        assert isinstance(embeddings[0], np.ndarray)
        assert embeddings[0].shape == (1536,)
        assert embeddings[0].dtype == np.float32

        # Verify cost tracking
        assert generator.total_tokens == 100
        assert generator.total_cost > 0

        # Verify API call
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-small"
        assert call_args.kwargs["input"] == chunks

    @patch("src.ingestion.embedding_generator.OpenAI")
    def test_generate_embeddings_batch_processing(self, mock_openai_class):
        """Test batch processing for >100 chunks."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response - return different embeddings for each batch
        def create_mock_response(model, input):
            embeddings = []
            for _ in range(len(input)):
                mock_emb = Mock()
                mock_emb.embedding = [0.1] * 1536
                embeddings.append(mock_emb)

            mock_response = Mock()
            mock_response.data = embeddings
            mock_response.usage.total_tokens = len(input) * 10
            return mock_response

        mock_client.embeddings.create.side_effect = create_mock_response

        # Test with 250 chunks (should create 3 batches: 100, 100, 50)
        generator = EmbeddingGenerator(api_key="test-key")
        chunks = [f"chunk {i}" for i in range(250)]
        embeddings = generator.generate_embeddings(chunks)

        # Verify
        assert len(embeddings) == 250
        assert all(e is not None for e in embeddings)

        # Verify 3 API calls were made
        assert mock_client.embeddings.create.call_count == 3

        # Verify batch sizes
        call_args_list = mock_client.embeddings.create.call_args_list
        assert len(call_args_list[0].kwargs["input"]) == 100
        assert len(call_args_list[1].kwargs["input"]) == 100
        assert len(call_args_list[2].kwargs["input"]) == 50

    @patch("src.ingestion.embedding_generator.OpenAI")
    @patch("time.sleep")
    def test_retry_logic_on_rate_limit(self, mock_sleep, mock_openai_class):
        """Test retry logic on rate limit error."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First call raises RateLimitError, second succeeds
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 10

        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            mock_response,
        ]

        # Test
        generator = EmbeddingGenerator(api_key="test-key")
        embeddings = generator.generate_embeddings(["test chunk"])

        # Verify
        assert len(embeddings) == 1
        assert embeddings[0] is not None

        # Verify retry happened
        assert mock_client.embeddings.create.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1 (first retry)

    @patch("src.ingestion.embedding_generator.OpenAI")
    @patch("time.sleep")
    def test_retry_logic_exponential_backoff(self, mock_sleep, mock_openai_class):
        """Test exponential backoff timing."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # All calls fail
        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit", response=Mock(), body=None),
            RateLimitError("Rate limit", response=Mock(), body=None),
            RateLimitError("Rate limit", response=Mock(), body=None),
        ]

        # Test
        generator = EmbeddingGenerator(api_key="test-key")
        embeddings = generator.generate_embeddings(["test chunk"])

        # Verify
        assert embeddings[0] is None  # Failed after max retries

        # Verify exponential backoff: 2^0=1, 2^1=2
        assert mock_sleep.call_count == 2
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2]  # 2^0, 2^1

    @patch("src.ingestion.embedding_generator.OpenAI")
    def test_authentication_error_raises_immediately(self, mock_openai_class):
        """Test authentication errors are not retried."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.embeddings.create.side_effect = AuthenticationError(
            "Invalid API key", response=Mock(), body=None
        )

        # Test
        generator = EmbeddingGenerator(api_key="test-key")

        with pytest.raises(ConfigurationError) as exc_info:
            generator.generate_embeddings(["test chunk"])

        assert "Invalid OpenAI API key" in str(exc_info.value)

        # Should only call once (no retries)
        assert mock_client.embeddings.create.call_count == 1

    @patch("src.ingestion.embedding_generator.OpenAI")
    @patch("time.sleep")
    def test_openai_error_retry(self, mock_sleep, mock_openai_class):
        """Test retry on generic OpenAI errors."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First call fails, second succeeds
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 10

        mock_client.embeddings.create.side_effect = [
            OpenAIError("API Error"),
            mock_response,
        ]

        # Test
        generator = EmbeddingGenerator(api_key="test-key")
        embeddings = generator.generate_embeddings(["test chunk"])

        # Verify success after retry
        assert embeddings[0] is not None
        assert mock_client.embeddings.create.call_count == 2

    @patch("src.ingestion.embedding_generator.OpenAI")
    def test_cost_tracking(self, mock_openai_class):
        """Test cost tracking accuracy."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 1000

        mock_client.embeddings.create.return_value = mock_response

        # Test
        generator = EmbeddingGenerator(api_key="test-key")
        generator.generate_embeddings(["test chunk"])

        # Verify cost tracking
        assert generator.total_tokens == 1000
        expected_cost = 1000 * (0.02 / 1_000_000)  # $0.00002
        assert abs(generator.total_cost - expected_cost) < 0.000001

        # Test cost summary
        summary = generator.get_cost_summary()
        assert summary["total_tokens"] == 1000
        assert abs(summary["total_cost_usd"] - expected_cost) < 0.0001
        assert summary["cost_per_1k_tokens"] == 0.00002

    @patch("src.ingestion.embedding_generator.OpenAI")
    def test_partial_failure_handling(self, mock_openai_class):
        """Test handling when some chunks fail."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First batch succeeds, second batch fails
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding] * 100
        mock_response.usage.total_tokens = 1000

        mock_client.embeddings.create.side_effect = [
            mock_response,  # First batch succeeds
            OpenAIError("API Error"),  # Second batch fails
            OpenAIError("API Error"),  # Retry 1 fails
            OpenAIError("API Error"),  # Retry 2 fails
        ]

        # Test with 150 chunks
        generator = EmbeddingGenerator(api_key="test-key")
        chunks = [f"chunk {i}" for i in range(150)]
        embeddings = generator.generate_embeddings(chunks)

        # Verify partial results
        assert len(embeddings) == 150
        # First 100 should succeed
        assert all(e is not None for e in embeddings[:100])
        # Last 50 should be None (failed)
        assert all(e is None for e in embeddings[100:])

    @patch("src.ingestion.embedding_generator.OpenAI")
    def test_invalid_chunks_filtered(self, mock_openai_class):
        """Test that invalid chunks are filtered and tracked correctly."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding, mock_embedding]
        mock_response.usage.total_tokens = 20

        mock_client.embeddings.create.return_value = mock_response

        # Test with mix of valid and invalid chunks
        generator = EmbeddingGenerator(api_key="test-key")
        chunks = ["valid 1", "", "valid 2", "   ", "valid 3"]
        embeddings = generator.generate_embeddings(chunks)

        # Should return None for invalid chunks at correct positions
        assert len(embeddings) == 5
        assert embeddings[0] is not None  # valid 1
        assert embeddings[1] is None  # empty
        assert embeddings[2] is not None  # valid 2
        assert embeddings[3] is None  # whitespace

        # The API should only be called once with valid chunks
        assert mock_client.embeddings.create.call_count == 1

    @patch("src.ingestion.embedding_generator.OpenAI")
    def test_rate_limiter_integration(self, mock_openai_class):
        """Test that rate limiter is called during generation."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 10

        mock_client.embeddings.create.return_value = mock_response

        # Test
        generator = EmbeddingGenerator(api_key="test-key", max_rpm=3000)
        generator.generate_embeddings(["test"])

        # Verify rate limiter tracked the request
        assert len(generator.rate_limiter.requests) == 1
