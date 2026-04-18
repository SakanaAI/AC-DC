"""
Unit tests for SimpleVectorDB (Vector Database for task similarity).

Tests core operations: add, get, delete, search, import/export.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import tempfile
import shutil

from tasks.simple_vectordb import SimpleVectorDB


class TestVectorDBCore:
    """Test basic vector database operations."""

    def test_initialization_with_mock_embedding(self, mock_embedding_client):
        """Test DB initialization with mocked embedding server."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tasks.simple_vectordb.OpenAI") as mock_openai:
                # Mock the embedding response for dimension detection
                mock_client = Mock()
                mock_embedding_data = Mock()
                mock_embedding_data.embedding = [0.1] * 384  # 384-dim vector
                mock_response = Mock()
                mock_response.data = [mock_embedding_data]
                mock_client.embeddings.create.return_value = mock_response
                mock_openai.return_value = mock_client

                db = SimpleVectorDB(
                    storage_path=temp_dir,
                    embedding_model_name="test-model",
                    embedding_vllm_url="http://localhost:8010/v1",
                    task_representation_vector_db="metadata",
                )

                # Verify initialization
                assert db.dimension == 384
                assert db.storage_path == temp_dir
                assert (Path(temp_dir) / "config.json").exists()

                # Verify directory structure
                assert (Path(temp_dir) / "vectors").exists()
                assert (Path(temp_dir) / "metadata").exists()
                assert (Path(temp_dir) / "raw_data").exists()
                assert (Path(temp_dir) / "index").exists()

    def test_add_sample_with_metadata_representation(
        self, vector_db_with_mock_embedding
    ):
        """Test adding sample with metadata-based embedding."""
        db = vector_db_with_mock_embedding

        content = "def test(): return True"
        metadata = {
            "task_name": "simple_test",
            "difficulty": "easy",
            "category": "logic",
        }

        sample_id = db.add_sample(content=content, metadata=metadata)

        # Verify sample was added
        assert sample_id is not None
        assert db.get_count() == 1

        # Verify file structure
        vectors_file = Path(db.vectors_path) / f"{sample_id}.npy"
        metadata_file = Path(db.metadata_path) / f"{sample_id}.json"
        content_file = Path(db.raw_data_path) / f"{sample_id}.py"
        index_file = Path(db.index_dir_path) / f"{sample_id}.json"

        assert vectors_file.exists()
        assert metadata_file.exists()
        assert content_file.exists()
        assert index_file.exists()

    def test_add_sample_with_custom_id(self, vector_db_with_mock_embedding):
        """Test adding sample with custom ID."""
        db = vector_db_with_mock_embedding

        custom_id = "task_123_example_0"
        content = "def custom(): pass"
        metadata = {"custom": True}

        sample_id = db.add_sample(
            content=content, metadata=metadata, custom_id=custom_id
        )

        # Verify custom ID is used (possibly sanitized)
        assert sample_id == custom_id.replace("/", "_")
        assert db.get_count() == 1

    def test_duplicate_id_prevention(self, vector_db_with_mock_embedding):
        """Test that duplicate IDs are rejected."""
        db = vector_db_with_mock_embedding

        sample_id = db.add_sample(
            content="def test1(): pass",
            metadata={"test": 1},
            custom_id="duplicate_test",
        )

        # Try to add with same ID
        with pytest.raises(ValueError, match="already exists"):
            db.add_sample(
                content="def test2(): pass",
                metadata={"test": 2},
                custom_id="duplicate_test",
            )

    def test_get_sample(self, vector_db_with_mock_embedding):
        """Test retrieving sample by ID."""
        db = vector_db_with_mock_embedding

        original_content = "def retrieve_me(): return 42"
        original_metadata = {"purpose": "retrieval_test"}

        sample_id = db.add_sample(
            content=original_content, metadata=original_metadata
        )

        # Retrieve the sample
        retrieved = db.get_sample(sample_id)

        assert retrieved is not None
        assert retrieved["id"] == sample_id
        assert retrieved["content"] == original_content
        assert retrieved["metadata"] == original_metadata
        assert "embedding" in retrieved
        assert isinstance(retrieved["embedding"], np.ndarray)

    def test_get_nonexistent_sample(self, vector_db_with_mock_embedding):
        """Test retrieving non-existent sample returns None."""
        db = vector_db_with_mock_embedding

        result = db.get_sample("nonexistent_id")
        assert result is None

    def test_delete_sample(self, vector_db_with_mock_embedding):
        """Test sample deletion and cleanup."""
        db = vector_db_with_mock_embedding

        sample_id = db.add_sample(
            content="def to_delete(): pass", metadata={"delete_me": True}
        )

        # Verify it exists
        assert db.get_count() == 1
        assert db.get_sample(sample_id) is not None

        # Delete it
        success = db.delete_sample(sample_id)
        assert success is True
        assert db.get_count() == 0
        assert db.get_sample(sample_id) is None

        # Verify files are cleaned up
        vectors_file = Path(db.vectors_path) / f"{sample_id}.npy"
        metadata_file = Path(db.metadata_path) / f"{sample_id}.json"
        content_file = Path(db.raw_data_path) / f"{sample_id}.py"
        index_file = Path(db.index_dir_path) / f"{sample_id}.json"

        assert not vectors_file.exists()
        assert not metadata_file.exists()
        assert not content_file.exists()
        assert not index_file.exists()

    def test_delete_nonexistent_sample(self, vector_db_with_mock_embedding):
        """Test deleting non-existent sample returns False."""
        db = vector_db_with_mock_embedding

        success = db.delete_sample("nonexistent_id")
        assert success is False

    def test_update_sample_metadata(self, vector_db_with_mock_embedding):
        """Test updating sample metadata."""
        db = vector_db_with_mock_embedding

        sample_id = db.add_sample(
            content="def test(): pass", metadata={"version": 1, "status": "draft"}
        )

        # Update metadata (merge mode)
        success = db.update_sample_metadata(
            sample_id, {"version": 2, "reviewed": True}, merge=True
        )

        assert success is True

        # Verify metadata was updated
        retrieved = db.get_sample(sample_id)
        assert retrieved["metadata"]["version"] == 2
        assert retrieved["metadata"]["status"] == "draft"  # Original preserved
        assert retrieved["metadata"]["reviewed"] is True  # New field added

    def test_update_sample_metadata_replace(self, vector_db_with_mock_embedding):
        """Test replacing sample metadata (merge=False)."""
        db = vector_db_with_mock_embedding

        sample_id = db.add_sample(
            content="def test(): pass", metadata={"old": "data", "keep": False}
        )

        # Replace metadata
        success = db.update_sample_metadata(
            sample_id, {"new": "data"}, merge=False
        )

        assert success is True

        # Verify metadata was replaced
        retrieved = db.get_sample(sample_id)
        assert retrieved["metadata"] == {"new": "data"}
        assert "old" not in retrieved["metadata"]

    def test_get_all_sample_ids(self, vector_db_with_mock_embedding):
        """Test retrieving all sample IDs."""
        db = vector_db_with_mock_embedding

        # Add multiple samples
        ids = []
        for i in range(5):
            sample_id = db.add_sample(
                content=f"def test_{i}(): pass", metadata={"index": i}
            )
            ids.append(sample_id)

        # Get all IDs
        all_ids = db.get_all_sample_ids()

        assert len(all_ids) == 5
        assert set(all_ids) == set(ids)

    def test_get_count(self, vector_db_with_mock_embedding):
        """Test counting samples in database."""
        db = vector_db_with_mock_embedding

        assert db.get_count() == 0

        # Add samples
        db.add_sample(content="def test1(): pass", metadata={})
        assert db.get_count() == 1

        db.add_sample(content="def test2(): pass", metadata={})
        assert db.get_count() == 2

        # Delete one
        ids = db.get_all_sample_ids()
        db.delete_sample(ids[0])
        assert db.get_count() == 1

    def test_sanitize_sample_id(self, vector_db_with_mock_embedding):
        """Test that sample IDs are sanitized (slashes replaced)."""
        db = vector_db_with_mock_embedding

        # ID with slashes
        sample_id = db.add_sample(
            content="def test(): pass",
            metadata={},
            custom_id="path/to/task",
        )

        # Verify slashes are replaced with underscores
        assert sample_id == "path_to_task"
        assert db.get_sample(sample_id) is not None


class TestVectorDBBatch:
    """Test batch operations."""

    def test_batch_add_samples(self, vector_db_with_mock_embedding):
        """Test adding multiple samples at once."""
        db = vector_db_with_mock_embedding

        samples = [
            {
                "content": f"def task_{i}(): pass",
                "metadata": {"task_id": i, "batch": True},
            }
            for i in range(10)
        ]

        added_ids = db.batch_add_samples(samples)

        assert len(added_ids) == 10
        assert db.get_count() == 10

    def test_batch_add_with_failures(self, vector_db_with_mock_embedding):
        """Test batch add continues on individual failures."""
        db = vector_db_with_mock_embedding

        # First add a sample to create a duplicate scenario
        db.add_sample(content="def test(): pass", metadata={}, custom_id="dup")

        samples = [
            {"content": "def task_1(): pass", "metadata": {"id": 1}},
            {
                "content": "def dup(): pass",
                "metadata": {"id": 2},
                "custom_id": "dup",
            },  # Will fail
            {"content": "def task_3(): pass", "metadata": {"id": 3}},
        ]

        added_ids = db.batch_add_samples(samples)

        # Should have added 2 out of 3 (duplicate fails)
        assert len(added_ids) == 2
        assert db.get_count() == 3  # 1 original + 2 new


class TestVectorDBExportImport:
    """Test database export and import functionality."""

    def test_export_database(self, vector_db_with_samples, tmp_path):
        """Test database export to zip."""
        db = vector_db_with_samples

        export_path = str(tmp_path / "exported_db.zip")
        result_path = db.export_database(export_path)

        assert result_path is not None
        assert Path(result_path).exists()
        assert result_path.endswith(".zip")

    def test_import_database_overwrite(self, tmp_path, mock_embedding_client):
        """Test import with merge=False (overwrite)."""
        # Create and populate first DB
        db1_path = tmp_path / "db1"
        db1_path.mkdir()

        with patch("tasks.simple_vectordb.OpenAI") as mock_openai:
            mock_openai.return_value = mock_embedding_client

            db1 = SimpleVectorDB(
                storage_path=str(db1_path),
                embedding_model_name="test-model",
                task_representation_vector_db="metadata",
                dimension=384,
            )

            # Add samples to db1
            sample_id_1 = db1.add_sample(
                content="def task1(): pass", metadata={"db": 1}
            )

            # Export db1
            export_path = str(tmp_path / "export.zip")
            db1.export_database(export_path)

            # Create second DB and import (overwrite mode)
            db2_path = tmp_path / "db2"
            db2_path.mkdir()

            db2 = SimpleVectorDB(
                storage_path=str(db2_path),
                embedding_model_name="test-model",
                task_representation_vector_db="metadata",
                dimension=384,
            )

            # Add different sample to db2
            db2.add_sample(content="def task2(): pass", metadata={"db": 2})

            # Import (overwrite)
            imported_count = db2.import_database(export_path, merge=False)

            # Verify db2 now has only db1's content
            assert imported_count == 1
            assert sample_id_1 in db2.get_all_sample_ids()

    def test_import_database_merge(self, tmp_path, mock_embedding_client):
        """Test import with merge=True."""
        # Create and populate first DB
        db1_path = tmp_path / "db1"
        db1_path.mkdir()

        with patch("tasks.simple_vectordb.OpenAI") as mock_openai:
            mock_openai.return_value = mock_embedding_client

            db1 = SimpleVectorDB(
                storage_path=str(db1_path),
                embedding_model_name="test-model",
                task_representation_vector_db="metadata",
                dimension=384,
            )

            sample_id_1 = db1.add_sample(
                content="def task1(): pass", metadata={"from": "db1"}
            )

            export_path = str(tmp_path / "export.zip")
            db1.export_database(export_path)

            # Create second DB
            db2_path = tmp_path / "db2"
            db2_path.mkdir()

            db2 = SimpleVectorDB(
                storage_path=str(db2_path),
                embedding_model_name="test-model",
                task_representation_vector_db="metadata",
                dimension=384,
            )

            sample_id_2 = db2.add_sample(
                content="def task2(): pass", metadata={"from": "db2"}
            )

            # Import (merge mode)
            imported_count = db2.import_database(export_path, merge=True)

            # Verify db2 has both samples
            assert imported_count == 2
            all_ids = db2.get_all_sample_ids()
            assert sample_id_1 in all_ids
            assert sample_id_2 in all_ids
