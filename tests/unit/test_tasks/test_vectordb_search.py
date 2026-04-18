"""
Unit tests for SimpleVectorDB search and similarity functionality.

Tests similarity search, threshold filtering, and ranking.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from tasks.simple_vectordb import SimpleVectorDB


class TestVectorDBSearch:
    """Test vector database search functionality."""

    def test_find_similar_by_metadata(self, vector_db_with_diverse_samples):
        """Test similarity search using metadata."""
        db = vector_db_with_diverse_samples

        # Search for logic tasks
        query_metadata = {"category": "logic", "difficulty": "medium"}

        results = db.find_similar(
            query=None, metadata=query_metadata, top_n=3, similarity_threshold=0.0
        )

        assert len(results) <= 3
        assert all("sample_id" in r for r in results)
        assert all("similarity" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("content" in r for r in results)

    def test_find_similar_by_content(self, vector_db_with_diverse_samples):
        """Test similarity search using content."""
        db = vector_db_with_diverse_samples

        # Create a DB with content-based representation
        # (This test assumes the fixture can be configured for content mode)
        query_text = "def fibonacci(n): return n"

        results = db.find_similar(
            query=query_text,
            metadata=None,
            top_n=5,
            similarity_threshold=0.0,
        )

        assert len(results) <= 5

    def test_similarity_threshold_filtering(
        self, vector_db_with_diverse_samples
    ):
        """Test that threshold filtering works."""
        db = vector_db_with_diverse_samples

        query_metadata = {"category": "test"}

        # High threshold should return fewer results
        results_high = db.find_similar(
            query=None,
            metadata=query_metadata,
            top_n=10,
            similarity_threshold=0.9,
        )

        # Low threshold should return more results
        results_low = db.find_similar(
            query=None,
            metadata=query_metadata,
            top_n=10,
            similarity_threshold=0.1,
        )

        assert len(results_high) <= len(results_low)

        # All results should meet threshold
        for result in results_high:
            assert result["similarity"] >= 0.9

    def test_similarity_scores_descending(self, vector_db_with_diverse_samples):
        """Test that results are sorted by similarity (descending)."""
        db = vector_db_with_diverse_samples

        query_metadata = {"category": "logic"}

        results = db.find_similar(
            query=None, metadata=query_metadata, top_n=5, similarity_threshold=0.0
        )

        # Verify descending order
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i + 1]["similarity"]

    def test_empty_database_search(self, vector_db_with_mock_embedding):
        """Test search on empty database."""
        db = vector_db_with_mock_embedding

        query_metadata = {"anything": "value"}

        results = db.find_similar(
            query=None, metadata=query_metadata, top_n=5
        )

        assert results == []

    def test_top_n_limiting(self, vector_db_with_many_samples):
        """Test that top_n parameter limits results."""
        db = vector_db_with_many_samples

        query_metadata = {"category": "common"}

        # Request only top 3
        results = db.find_similar(
            query=None,
            metadata=query_metadata,
            top_n=3,
            similarity_threshold=0.0,
        )

        assert len(results) <= 3

    def test_find_similar_with_precomputed_embedding(
        self, vector_db_with_diverse_samples
    ):
        """Test similarity search with precomputed embedding."""
        db = vector_db_with_diverse_samples

        # Create a query embedding (should match dimension)
        query_embedding = np.random.rand(db.dimension).astype(np.float32)

        results = db.find_similar(
            query=None,
            metadata=None,
            query_embedding=query_embedding,
            top_n=5,
        )

        assert len(results) <= 5

    def test_find_similar_no_query_raises_error(
        self, vector_db_with_mock_embedding
    ):
        """Test that find_similar raises error without query or embedding."""
        db = vector_db_with_mock_embedding

        with pytest.raises(ValueError, match="Either query_text, query_embedding, or metadata must be provided"):
            db.find_similar(
                query=None, metadata=None, query_embedding=None, top_n=5
            )

    def test_similarity_range(self, vector_db_with_diverse_samples):
        """Test that similarity scores are in valid range [0, 1]."""
        db = vector_db_with_diverse_samples

        query_metadata = {"category": "test"}

        results = db.find_similar(
            query=None, metadata=query_metadata, top_n=10
        )

        for result in results:
            assert 0.0 <= result["similarity"] <= 1.0

    def test_find_similar_returns_complete_data(
        self, vector_db_with_diverse_samples
    ):
        """Test that find_similar returns complete sample data."""
        db = vector_db_with_diverse_samples

        query_metadata = {"category": "logic"}

        results = db.find_similar(
            query=None, metadata=query_metadata, top_n=1
        )

        if results:
            result = results[0]

            # Verify all expected fields are present
            assert "sample_id" in result
            assert "similarity" in result
            assert "metadata" in result
            assert "content" in result

            # Verify metadata is dict
            assert isinstance(result["metadata"], dict)

            # Verify content is string
            assert isinstance(result["content"], str)


class TestVectorDBEmbedding:
    """Test embedding generation."""

    def test_embed_text_with_metadata_mode(self, vector_db_with_mock_embedding):
        """Test embedding generation in metadata mode."""
        db = vector_db_with_mock_embedding

        # The DB should be in metadata mode
        assert db.task_representation_vector_db == "metadata"

        # Embed some metadata text
        text = "category: logic\ndifficulty: hard\ntask_id: 123"
        embedding = db.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (db.dimension,)
        assert embedding.dtype == np.float32

    def test_embed_text_returns_zero_on_error(self, tmp_path):
        """Test that embed_text returns zero vector on error."""
        with patch("tasks.simple_vectordb.OpenAI") as mock_openai:
            # Mock successful init but failing embed
            mock_client_init = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 384
            mock_response_init = Mock()
            mock_response_init.data = [mock_embedding_data]
            mock_client_init.embeddings.create.return_value = (
                mock_response_init
            )

            # First call succeeds (for init), subsequent calls fail
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_client_init
                else:
                    # Create client that raises exception on embed
                    mock_client_fail = Mock()
                    mock_client_fail.embeddings.create.side_effect = Exception(
                        "Embedding server error"
                    )
                    return mock_client_fail

            mock_openai.side_effect = side_effect

            db = SimpleVectorDB(
                storage_path=str(tmp_path),
                embedding_model_name="test-model",
                task_representation_vector_db="metadata",
            )

            # Try to embed - should return zero vector
            embedding = db.embed_text("test text")

            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (db.dimension,)
            assert np.all(embedding == 0.0)


class TestVectorDBConfigPersistence:
    """Test configuration persistence and loading."""

    def test_config_saved_on_init(self, tmp_path, mock_embedding_client):
        """Test that config is saved during initialization."""
        with patch("tasks.simple_vectordb.OpenAI") as mock_openai:
            mock_openai.return_value = mock_embedding_client

            db = SimpleVectorDB(
                storage_path=str(tmp_path),
                embedding_model_name="test-model-v1",
                task_representation_vector_db="metadata",
                dimension=384,
            )

            config_path = tmp_path / "config.json"
            assert config_path.exists()

            # Load and verify config
            with open(config_path) as f:
                config = pytest.importorskip("json").load(f)

            assert config["model_info"]["name"] == "test-model-v1"
            assert config["model_info"]["dimension"] == 384

    def test_config_loaded_on_subsequent_init(
        self, tmp_path, mock_embedding_client
    ):
        """Test that config is loaded if it already exists."""
        with patch("tasks.simple_vectordb.OpenAI") as mock_openai:
            mock_openai.return_value = mock_embedding_client

            # First init - creates config
            db1 = SimpleVectorDB(
                storage_path=str(tmp_path),
                embedding_model_name="test-model",
                task_representation_vector_db="metadata",
                dimension=384,
            )

            # Second init - should load config
            db2 = SimpleVectorDB(
                storage_path=str(tmp_path),
                embedding_model_name="test-model",
                task_representation_vector_db="metadata",
            )

            # Dimension should be loaded from config
            assert db2.dimension == 384
