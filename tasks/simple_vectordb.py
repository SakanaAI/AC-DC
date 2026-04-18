"""
Semantic Vector Database Interface (Lock-Free, Directory Index)

This module provides a comprehensive interface for embedding, storing,
retrieving, and deleting text samples in a vector database using a
directory-based index to avoid multiprocessing locks.
"""

import json
import os
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any
import shutil
import tempfile
from datetime import datetime
import logging
import zipfile

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


# Helper function to import time module safely (if needed, though datetime is used now)
# def import_time_module():
#     import time
#     return time.time() # Or use datetime.now().isoformat() directly


class SimpleVectorDB:
    """
    A class for handling text samples with semantic embeddings,
    providing storage, retrieval, and deletion functionality using
    a lock-free, directory-based index.
    """

    INDEX_MARKER_EXT = ".json"
    CONFIG_FILENAME = "config.json"

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_vllm_url: str = "http://localhost:8010/v1",
        storage_path: str = "./vector_db",
        dimension: Optional[int] = None,
        max_seq_length: Optional[
            int
        ] = None,  # Note: max_seq_length might not be applicable to OpenAI client
        task_representation_vector_db: str = "metadata",
    ):
        """
        Initialize the SimpleVectorDB.

        Args:
            embedding_model_name: Name of the sentence-transformers model or OpenAI model ID.
            embedding_vllm_url: Base URL for the vLLM OpenAI-compatible server.
            storage_path: Directory to store embedded samples and index.
            dimension: Optional dimension of embedding vectors, auto-detected if None.
            max_seq_length: Optional maximum sequence length (informational, may not be enforced by client).
            task_representation_vector_db: Whether to use 'metadata' or 'content' for embedding.
        """
        self.logger = logging.getLogger("SimpleVectorDB")

        # Store embedding model details
        self.embedding_model_name = (
            embedding_model_name.split("/")[-1]
            if "/" in embedding_model_name
            else embedding_model_name
        )
        self.embedding_vllm_url = embedding_vllm_url

        # Validate task representation
        self.task_representation_vector_db = task_representation_vector_db
        if self.task_representation_vector_db not in [
            "metadata",
            "content",
            "all",
        ]:
            raise ValueError(
                f"Invalid task_representation_vector_db: {self.task_representation_vector_db}. Must be 'metadata' or 'content'."
            )

        if max_seq_length is not None:
            self.logger.warning(
                "max_seq_length is set but might not be applicable to the OpenAI client."
            )

        self.storage_path = storage_path
        self.config_path = os.path.join(storage_path, self.CONFIG_FILENAME)

        # Initialize storage directories
        self.vectors_path = os.path.join(storage_path, "vectors")
        self.metadata_path = os.path.join(storage_path, "metadata")
        self.raw_data_path = os.path.join(storage_path, "raw_data")
        self.index_dir_path = os.path.join(
            storage_path, "index"
        )  # Directory for index markers

        os.makedirs(self.vectors_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.index_dir_path, exist_ok=True)

        # Load or initialize configuration (model info, dimension)
        loaded_dimension = None
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                # Validate config
                if (
                    config.get("model_info", {}).get("name")
                    != self.embedding_model_name
                ):
                    self.logger.warning(
                        f"Config model name '{config.get('model_info', {}).get('name')}' "
                        f"differs from requested '{self.embedding_model_name}'. Using requested."
                    )
                    # Potentially raise error or force re-embedding if strict matching is needed
                loaded_dimension = config.get("model_info", {}).get("dimension")
                if dimension is not None and loaded_dimension != dimension:
                    self.logger.warning(
                        f"Provided dimension {dimension} differs from stored dimension {loaded_dimension}. Using stored."
                    )
                    dimension = loaded_dimension  # Prioritize stored dimension
                elif dimension is None and loaded_dimension is not None:
                    dimension = loaded_dimension  # Use stored dimension
                self.logger.info(f"Loaded config from {self.config_path}")

            except json.JSONDecodeError:
                self.logger.error(
                    f"Error decoding JSON from {self.config_path}. Reinitializing config."
                )
            except Exception as e:
                self.logger.error(
                    f"Error loading config {self.config_path}: {e}. Reinitializing config."
                )

        # Auto-detect dimension if still needed
        if dimension is None:
            if loaded_dimension:
                dimension = loaded_dimension
            else:
                self.logger.info(
                    "Attempting to auto-detect embedding dimension..."
                )
                try:
                    # Create client locally for safety in case of concurrent init attempts (though unlikely)
                    temp_client = OpenAI(
                        base_url=self.embedding_vllm_url, api_key="EMPTY"
                    )
                    test_embedding_data = (
                        temp_client.embeddings.create(
                            input="test", model=self.embedding_model_name
                        )
                        .data[0]
                        .embedding
                    )
                    dimension = len(test_embedding_data)
                    self.logger.info(f"Auto-detected dimension: {dimension}")
                    del temp_client
                except Exception as e:
                    self.logger.error(
                        f"Failed to auto-detect embedding dimension: {e}. Please provide dimension manually."
                    )
                    raise ValueError(
                        "Could not determine embedding dimension automatically."
                    ) from e

        self.dimension = dimension

        # Save config if it didn't exist or was invalid/updated
        if (
            not os.path.exists(self.config_path)
            or loaded_dimension != self.dimension
        ):
            self._save_config()

    def _save_config(self) -> None:
        """Save the model configuration to disk."""
        config = {
            "model_info": {
                "name": self.embedding_model_name,
                "dimension": self.dimension,
            }
        }
        temp_path = None
        try:
            # Use atomic write pattern
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.storage_path, prefix="config_"
            )
            with os.fdopen(temp_fd, "w") as tmp_f:
                json.dump(config, tmp_f, indent=2)
            shutil.move(temp_path, self.config_path)
            temp_path = None
            self.logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            # Clean up temp file if move failed
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as rm_e:
                    self.logger.error(
                        f"Error removing temporary config file {temp_path}: {rm_e}"
                    )

    def _get_index_marker_path(self, sample_id: str) -> str:
        """Get the path for a sample's index marker file."""
        # Basic sanitization for sample_id to prevent path traversal issues
        sample_id = self._sanitize_sample_id(sample_id)
        safe_sample_id = os.path.basename(sample_id)
        if safe_sample_id != sample_id:
            self.logger.warning(
                f"Sample ID '{sample_id}' contained path elements, using '{safe_sample_id}'."
            )
            # Consider raising an error if strict ID format is required
        return os.path.join(
            self.index_dir_path, f"{safe_sample_id}{self.INDEX_MARKER_EXT}"
        )

    def _generate_sample_id(self) -> str:
        """Generate a unique ID for a new sample."""
        return str(uuid.uuid4())

    def _sanitize_sample_id(self, sample_id: str) -> str:
        """Sanitize a sample ID by replacing slashes with hyphens."""
        return sample_id.replace("/", "_")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for the given text using the configured OpenAI client.

        Args:
            text: The text to embed.

        Returns:
            A numpy array containing the embedding vector.
        """
        try:
            # Create client locally within the method for process safety
            local_client = OpenAI(
                base_url=self.embedding_vllm_url, api_key="EMPTY"
            )
            embedding_data = (
                local_client.embeddings.create(
                    input=text,
                    model=self.embedding_model_name,
                )
                .data[0]
                .embedding
            )
            return np.array(embedding_data, dtype=np.float32)
        except Exception as e:
            self.logger.error(
                f"Error generating embedding for text: '{text[:100]}...': {e}",
                exc_info=True,
            )
            # Return a zero vector or raise, depending on desired behavior
            return np.zeros(
                self.dimension, dtype=np.float32 # type: ignore
            )

    def add_sample(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        custom_id: Optional[str] = None,
        precomputed_embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Add a new sample to the database.

        Args:
            content: The raw text content to store and embed.
            metadata: Optional dict of metadata associated with the sample.
            custom_id: Optional custom ID for the sample.
            precomputed_embedding: Optional precomputed embedding vector.

        Returns:
            The ID of the stored sample.

        Raises:
            ValueError: If a sample with the given ID already exists or if embedding dimensions mismatch.
            IOError: If saving data files fails.
        """
        sample_id = custom_id if custom_id else self._generate_sample_id()
        sample_id = self._sanitize_sample_id(sample_id)
        index_marker_path = self._get_index_marker_path(sample_id)

        # Check if ID already exists using the index marker file (atomic check)
        try:
            # Attempt to open the marker file exclusively for creation
            # This provides a more atomic check than os.path.exists followed by create
            fd = os.open(
                index_marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            # If successful, immediately close it; we'll write the timestamp later
            os.close(fd)
            # Remove the empty file for now; it will be created after data saving
            os.remove(index_marker_path)
        except FileExistsError:
            raise ValueError(
                f"Sample with ID {sample_id} already exists (index marker found)."
            )
        except Exception as e:
            # Handle other potential OS errors during the check
            self.logger.error(
                f"Error checking/creating index marker for {sample_id}: {e}"
            )
            raise IOError(
                f"Failed to check/create index marker for {sample_id}"
            ) from e

        # Create embedding if not provided
        if precomputed_embedding is None:
            if self.task_representation_vector_db == "metadata":
                if metadata is None:
                    self.logger.warning(
                        f"Metadata is None for sample {sample_id}, but task_representation is 'metadata'. Embedding empty string."
                    )
                    content_for_embedding = ""
                else:
                    # Simple string representation of metadata for embedding
                    content_for_embedding = "\n".join(
                        f"{k}: {v}" for k, v in sorted(metadata.items())
                    )
            elif self.task_representation_vector_db == "content":
                content_for_embedding = content
            elif self.task_representation_vector_db == "all":
                task_metadata = ""
                if metadata is not None:
                    for key, value in metadata.items():
                        task_metadata += f"{key}: {value}\n"
                content_for_embedding = f"Task description: \n{task_metadata}\nTask code: \n{content}"
            else:  # Should not happen due to assert in __init__
                raise ValueError(
                    f"Invalid task_representation_vector_db: {self.task_representation_vector_db}"
                )

            embedding = self.embed_text(content_for_embedding)
        else:
            embedding = np.array(precomputed_embedding, dtype=np.float32)
            if embedding.shape != (self.dimension,):
                raise ValueError(
                    f"Embedding dimension mismatch for ID {sample_id}: expected ({self.dimension},), "
                    f"got {embedding.shape}"
                )

        # Prepare metadata
        if metadata is None:
            metadata = {}

        # Define file paths
        vector_file = os.path.join(self.vectors_path, f"{sample_id}.npy")
        metadata_file = os.path.join(self.metadata_path, f"{sample_id}.json")
        content_file = os.path.join(self.raw_data_path, f"{sample_id}.py")

        # Save embedding, metadata, and raw content
        # Attempt to save all data files before creating the index marker
        temp_path_meta = None
        temp_path_cont = None
        try:
            np.save(vector_file, embedding)
            # Use atomic write for metadata JSON
            temp_fd_meta, temp_path_meta = tempfile.mkstemp(
                dir=self.metadata_path, prefix=f"{sample_id}_meta_"
            )
            with os.fdopen(temp_fd_meta, "w") as tmp_f_meta:
                json.dump(metadata, tmp_f_meta, indent=2)
            shutil.move(temp_path_meta, metadata_file)
            temp_path_meta = None  # Successfully moved, no cleanup needed

            # Use atomic write for content file
            temp_fd_cont, temp_path_cont = tempfile.mkstemp(
                dir=self.raw_data_path, prefix=f"{sample_id}_cont_"
            )
            with os.fdopen(temp_fd_cont, "w") as tmp_f_cont:
                tmp_f_cont.write(content)
            shutil.move(temp_path_cont, content_file)
            temp_path_cont = None  # Successfully moved, no cleanup needed

        except Exception as e:
            self.logger.error(
                f"Error saving data files for sample {sample_id}: {e}",
                exc_info=True,
            )
            # Cleanup partially written files
            for f_path in [vector_file, metadata_file, content_file]:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError as rm_e:
                        self.logger.error(
                            f"Error cleaning up file {f_path} after save error: {rm_e}"
                        )
            # Also cleanup temp files if they exist
            if temp_path_meta and os.path.exists(temp_path_meta):
                try:
                    os.remove(temp_path_meta)
                except OSError:
                    pass
            if temp_path_cont and os.path.exists(temp_path_cont):
                try:
                    os.remove(temp_path_cont)
                except OSError:
                    pass
            raise IOError(
                f"Failed to save data files for sample {sample_id}"
            ) from e

        # Create the index marker file to signify completion
        try:
            with open(index_marker_path, "w") as f:
                json.dump(
                    {
                        "id": sample_id,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self.logger.error(
                f"CRITICAL: Failed to create index marker {index_marker_path} after saving data for {sample_id}: {e}",
                exc_info=True,
            )
            # Data files exist but sample is not indexed. Requires manual intervention or cleanup logic.
            # Attempt cleanup of data files again
            for f_path in [vector_file, metadata_file, content_file]:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except OSError:
                        pass  # Ignore cleanup errors here
            raise IOError(
                f"Failed to create index marker for sample {sample_id} after saving data."
            ) from e

        self.logger.debug(f"Added sample {sample_id}")
        return sample_id

    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a sample by ID.

        Args:
            sample_id: The ID of the sample to retrieve.

        Returns:
            Dict containing the sample data, embedding, and metadata, or None if not found.
        """

        sample_id = self._sanitize_sample_id(sample_id)
        index_marker_path = self._get_index_marker_path(sample_id)

        # Check index marker first
        if not os.path.exists(index_marker_path):
            self.logger.debug(
                f"Sample {sample_id} not found (no index marker)."
            )
            return None

        try:
            # Define file paths
            embedding_path = os.path.join(self.vectors_path, f"{sample_id}.npy")
            metadata_path = os.path.join(
                self.metadata_path, f"{sample_id}.json"
            )
            content_path = os.path.join(self.raw_data_path, f"{sample_id}.py")

            # Check data file existence for robustness
            if not all(
                os.path.exists(p)
                for p in [embedding_path, metadata_path, content_path]
            ):
                self.logger.error(
                    f"Index inconsistency: Index marker exists for {sample_id}, but one or more data files are missing."
                )
                # Optionally try to delete the orphaned index marker here
                # try: os.remove(index_marker_path) except OSError: pass
                return None

            # Load data
            embedding = np.load(embedding_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            with open(content_path, "r") as f:
                content = f.read()

            # Optionally load timestamp from marker file if stored/needed
            timestamp = None
            try:
                with open(index_marker_path, "r") as f:
                    timestamp = f.read().strip()
            except Exception:
                self.logger.warning(
                    f"Could not read timestamp from marker file for {sample_id}"
                )

            return {
                "id": sample_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
                "timestamp": timestamp,  # Include timestamp if read
            }
        except FileNotFoundError as e:
            # Should be caught by initial checks, but handle defensively
            self.logger.error(
                f"File not found during get_sample for indexed sample {sample_id}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Error retrieving sample {sample_id}: {e}", exc_info=True
            )
            return None

    def delete_sample(self, sample_id: str) -> bool:
        """
        Delete a sample from the database.

        Args:
            sample_id: The ID of the sample to delete.

        Returns:
            True if the sample index marker was found and deleted, False otherwise.
            Note: File deletion errors are logged but don't change the return value if the index marker is handled.
        """
        sample_id = self._sanitize_sample_id(sample_id)
        index_marker_path = self._get_index_marker_path(sample_id)

        # Check if index marker exists first
        if not os.path.exists(index_marker_path):
            self.logger.debug(
                f"Sample {sample_id} not found for deletion (no index marker)."
            )
            return False

        # Delete data files first
        files_deleted_successfully = True
        for path_attr, ext in [
            ("vectors_path", "npy"),
            ("metadata_path", "json"),
            ("raw_data_path", "py"),
        ]:
            # Construct path using the original sample_id, not the potentially sanitized one used for index marker
            file_path = os.path.join(
                getattr(self, path_attr), f"{sample_id}.{ext}"
            )
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                self.logger.error(f"Error deleting data file {file_path}: {e}")
                files_deleted_successfully = False  # Log error but continue

        # Delete the index marker file last (using the potentially sanitized path)
        try:
            os.remove(index_marker_path)
            self.logger.debug(f"Deleted sample {sample_id}")
            return True  # Successfully removed from index
        except OSError as e:
            self.logger.error(
                f"Error deleting index marker file {index_marker_path}: {e}"
            )
            # Sample might be in an inconsistent state (data deleted, but still indexed)
            return False  # Indicate index deletion failed

    def update_sample_metadata(
        self, sample_id: str, metadata: Dict[str, Any], merge: bool = True
    ) -> bool:
        """
        Update metadata for an existing sample. Re-embedding is NOT done automatically.

        Args:
            sample_id: The ID of the sample to update.
            metadata: New metadata dict.
            merge: If True, merge with existing metadata; if False, replace.

        Returns:
            True if the metadata was updated, False if the sample wasn't found (no index marker).
        """
        sample_id = self._sanitize_sample_id(sample_id)
        index_marker_path = self._get_index_marker_path(sample_id)
        if not os.path.exists(index_marker_path):
            self.logger.warning(
                f"Cannot update metadata for non-existent sample {sample_id}."
            )
            return False

        metadata_path = os.path.join(self.metadata_path, f"{sample_id}.json")
        metadata_to_write = metadata

        if merge:
            try:
                # Read existing metadata if merging
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        existing = json.load(f)
                    existing.update(metadata)  # Update existing dict
                    metadata_to_write = existing
                else:
                    self.logger.warning(
                        f"Metadata file {metadata_path} not found for merging, writing new metadata anyway."
                    )
            except Exception as e:
                self.logger.error(
                    f"Error reading existing metadata for merge ({sample_id}): {e}. Overwriting."
                )
                # Fallback to overwrite if read fails

        # Write the new/merged metadata using atomic write
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.metadata_path, prefix=f"{sample_id}_meta_"
            )
            with os.fdopen(temp_fd, "w") as tmp_f:
                json.dump(metadata_to_write, tmp_f, indent=2)
            shutil.move(temp_path, metadata_path)
            temp_path = None  # Successfully moved, no cleanup needed
            self.logger.debug(f"Updated metadata for sample {sample_id}")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to write updated metadata for {sample_id}: {e}"
            )
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as rm_e:
                    self.logger.error(
                        f"Error removing temporary metadata file {temp_path}: {rm_e}"
                    )
            return False

    def find_similar(
        self,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_n: int = 5,
        similarity_threshold: Optional[float] = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find samples with embeddings similar to the query text or embedding.

        Args:
            query: Text to embed and use for querying.
            metadata: Metadata to use for querying.
            query_embedding: Precomputed embedding to use for querying.
            top_n: Maximum number of similar samples to return.
            similarity_threshold: Minimum cosine similarity score to include.

        Returns:
            List of dicts: each containing {"sample_id": str, "similarity": float, "metadata": dict, "content": str}.

        Raises:
            ValueError: If neither query_text nor query_embedding is provided.
        """
        if similarity_threshold is None:
            similarity_threshold = 0.0
        # Validate that at least one query method is provided
        if query_embedding is None and query is None and metadata is None:
            raise ValueError(
                "Either query_text, query_embedding, or metadata must be provided"
            )
        if query_embedding is None:
            if self.task_representation_vector_db == "metadata":
                text_to_embed = ""
                if metadata is None:
                    pass
                else:
                    for key, value in metadata.items():
                        text_to_embed += f"{key}: {value}\n"
            elif self.task_representation_vector_db == "content":
                text_to_embed = query or ""
            elif self.task_representation_vector_db == "all":
                task_metadata = ""
                if metadata is not None:
                    for key, value in metadata.items():
                        task_metadata += f"{key}: {value}\n"
                text_to_embed = (
                    f"Task description: \n{task_metadata}\nTask code: \n{query or ''}"
                )
            else:
                raise ValueError(
                    f"Invalid task representation: {self.task_representation_vector_db}"
                )
            query_embedding = self.embed_text(text_to_embed)

        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(
            1, -1
        )  # Ensure 2D for cosine_similarity

        all_sample_ids = self.get_all_sample_ids()
        if not all_sample_ids:
            return []

        return_items = []
        for sample_id in all_sample_ids:
            # Construct paths using the original sample_id
            embedding_path = os.path.join(self.vectors_path, f"{sample_id}.npy")
            metadata_path = os.path.join(
                self.metadata_path, f"{sample_id}.json"
            )
            content_path = os.path.join(self.raw_data_path, f"{sample_id}.py")

            # Check embedding exists (index guarantees it should, but check defensively)
            if not os.path.exists(embedding_path):
                self.logger.warning(
                    f"Skipping sample {sample_id} in find_similar: embedding file missing despite index marker."
                )
                continue

            try:
                sample_embedding = np.load(embedding_path).reshape(
                    1, -1
                )  # Ensure 2D
                similarity = cosine_similarity(
                    query_embedding, sample_embedding
                )[0][0]

                if similarity >= similarity_threshold:
                    # Load metadata only if similarity meets threshold (or no threshold)
                    metadata = {}
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                        except Exception as meta_e:
                            self.logger.error(
                                f"Error loading metadata for {sample_id} during similarity search: {meta_e}"
                            )
                    else:
                        self.logger.warning(
                            f"Metadata file missing for sample {sample_id} during similarity search."
                        )

                    content = ""
                    if os.path.exists(content_path):
                        try:
                            with open(content_path, "r") as f:
                                content = f.read()
                        except Exception as content_e:
                            self.logger.error(
                                f"Error loading content for {sample_id} during similarity search: {content_e}"
                            )

                    return_items.append(
                        {
                            "sample_id": sample_id,
                            "similarity": float(similarity),
                            "metadata": metadata,
                            "content": content,
                        }
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing sample {sample_id} during find_similar: {e}"
                )
                continue

        # Sort by similarity score (descending)
        return_items.sort(key=lambda item: item["similarity"], reverse=True)

        return return_items[:top_n]

    def batch_add_samples(self, samples: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple samples in a batch.

        Args:
            samples: List of dicts, each containing 'content', 'metadata' (optional),
                     'custom_id' (optional), 'precomputed_embedding' (optional).

        Returns:
            List of IDs of the added samples. Errors are logged, failed samples are skipped.
        """
        added_ids = []
        for i, sample_data in enumerate(samples):
            try:
                sample_id = self.add_sample(
                    content=sample_data["content"],
                    metadata=sample_data.get("metadata"),
                    custom_id=sample_data.get("custom_id"),
                    precomputed_embedding=sample_data.get(
                        "precomputed_embedding"
                    ),
                )
                added_ids.append(sample_id)
            except (ValueError, IOError) as e:
                self.logger.error(
                    f"Failed to add sample {i} in batch (ID: {sample_data.get('custom_id', 'N/A')}): {e}"
                )
            except Exception as e:
                self.logger.error(
                    f"Unexpected error adding sample {i} in batch (ID: {sample_data.get('custom_id', 'N/A')}): {e}",
                    exc_info=True,
                )

        return added_ids

    def get_all_sample_ids(self) -> List[str]:
        """
        Get a list of all sample IDs currently in the database index.

        Returns:
            List of sample IDs (original IDs, not sanitized marker filenames).
        """
        try:
            # List files and remove the marker extension to get original IDs
            ids = [
                f[: -len(self.INDEX_MARKER_EXT)]
                for f in os.listdir(self.index_dir_path)
                if f.endswith(self.INDEX_MARKER_EXT)
                and os.path.isfile(os.path.join(self.index_dir_path, f))
            ]
            return ids
        except OSError as e:
            self.logger.error(
                f"Error listing index directory {self.index_dir_path}: {e}"
            )
            return []

    def get_count(self) -> int:
        """Return the number of samples currently in the database index."""
        # Simply count the number of marker files
        try:
            count = len(
                [
                    f
                    for f in os.listdir(self.index_dir_path)
                    if f.endswith(self.INDEX_MARKER_EXT)
                    and os.path.isfile(os.path.join(self.index_dir_path, f))
                ]
            )
            return count
        except OSError as e:
            self.logger.error(
                f"Error counting index files in {self.index_dir_path}: {e}"
            )
            return 0  # Return 0 if directory cannot be read

    def export_database(self, output_path: str) -> Optional[str]:
        """
        Export the entire database (config, vectors, metadata, raw_data, index) to a zip file.

        Args:
            output_path: Path to save the zip file (e.g., '/path/to/db_export.zip').

        Returns:
            The absolute path to the created zip file, or None if export failed.
        """
        if not output_path.endswith(".zip"):
            output_path += ".zip"

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.logger.info(
            f"Exporting database from {self.storage_path} to {output_path}..."
        )

        try:
            with zipfile.ZipFile(
                output_path, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                # Add config file
                if os.path.exists(self.config_path):
                    zipf.write(self.config_path, self.CONFIG_FILENAME)
                    self.logger.debug(
                        f"Added {self.CONFIG_FILENAME} to export."
                    )
                else:
                    self.logger.warning(
                        f"Config file {self.config_path} not found for export."
                    )

                # Add data directories recursively
                for dir_name in ["vectors", "metadata", "raw_data", "index"]:
                    dir_path = os.path.join(self.storage_path, dir_name)
                    if os.path.isdir(dir_path):
                        for root, _, files in os.walk(dir_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                # arcname is the path inside the zip file
                                arcname = os.path.relpath(
                                    file_path, self.storage_path
                                )
                                zipf.write(file_path, arcname)
                        self.logger.debug(
                            f"Added '{dir_name}' directory contents to export."
                        )
                    else:
                        self.logger.warning(
                            f"Directory '{dir_name}' not found at {dir_path} for export."
                        )

            abs_output_path = os.path.abspath(output_path)
            self.logger.info(
                f"Database successfully exported to {abs_output_path}"
            )
            return abs_output_path
        except Exception as e:
            self.logger.error(
                f"Failed to export database to {output_path}: {e}",
                exc_info=True,
            )
            # Clean up potentially incomplete zip file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError as rm_e:
                    self.logger.error(
                        f"Error removing incomplete export file {output_path}: {rm_e}"
                    )
            return None

    def import_database(self, zip_path: str, merge: bool = False) -> int:
        """
        Import a database from a zip file created by export_database.

        Args:
            zip_path: Path to the zip file to import.
            merge: If True, merge with existing data (imported samples overwrite existing ones with the same ID).
                   If False (default), clear the existing database before importing.

        Returns:
            The number of samples successfully indexed after import.

        Raises:
            FileNotFoundError: If the zip_path does not exist.
            zipfile.BadZipFile: If the zip file is invalid.
            ValueError: If the imported config conflicts significantly (e.g., dimension mismatch) and merge=False.
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Import file not found: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            raise zipfile.BadZipFile(
                f"Import file is not a valid zip file: {zip_path}"
            )

        self.logger.info(
            f"Importing database from {zip_path} into {self.storage_path} (merge={merge})..."
        )

        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory(prefix="vectordb_import_") as temp_dir:
            try:
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    # Basic check for required directories in zip
                    required_dirs = {
                        "vectors/",
                        "metadata/",
                        "raw_data/",
                        "index/",
                    }
                    zip_contents = set(
                        item.filename for item in zipf.infolist()
                    )
                    # Check if top-level directories exist (adjust if structure varies)
                    # This is a basic check; more robust validation might be needed
                    # if not required_dirs.issubset(zip_contents):
                    #      self.logger.warning(f"Import archive {zip_path} might be missing expected directories.")

                    zipf.extractall(temp_dir)
                self.logger.debug(f"Extracted import file to {temp_dir}")
            except Exception as e:
                self.logger.error(f"Failed to extract zip file {zip_path}: {e}")
                raise IOError(f"Failed to extract import archive: {e}") from e

            # --- Config Handling ---
            imported_config_path = os.path.join(temp_dir, self.CONFIG_FILENAME)
            imported_dimension = None
            imported_model_name = None
            config_needs_saving = False
            if os.path.exists(imported_config_path):
                try:
                    with open(imported_config_path, "r") as f:
                        imported_config = json.load(f)
                    imported_dimension = imported_config.get(
                        "model_info", {}
                    ).get("dimension")
                    imported_model_name = imported_config.get(
                        "model_info", {}
                    ).get("name")
                    self.logger.info(
                        f"Found config in import: model={imported_model_name}, dim={imported_dimension}"
                    )

                    # Validation against current DB config
                    if not merge:
                        # If not merging, the imported config defines the new DB state
                        if self.dimension != imported_dimension:
                            self.logger.warning(
                                f"Imported dimension ({imported_dimension}) differs from current ({self.dimension}). Adopting imported dimension."
                            )
                            self.dimension = imported_dimension
                            config_needs_saving = True
                        if self.embedding_model_name != imported_model_name:
                            self.logger.warning(
                                f"Imported model name ({imported_model_name}) differs from current ({self.embedding_model_name}). Adopting imported name."
                            )
                            self.embedding_model_name = imported_model_name
                            config_needs_saving = True
                    else:
                        # If merging, dimensions MUST match
                        if self.dimension != imported_dimension:
                            raise ValueError(
                                f"Cannot merge databases with different embedding dimensions: "
                                f"Current={self.dimension}, Imported={imported_dimension}"
                            )
                        if self.embedding_model_name != imported_model_name:
                            self.logger.warning(
                                f"Merging databases with different model names: Current='{self.embedding_model_name}', Imported='{imported_model_name}'. Ensure compatibility."
                            )

                except Exception as e:
                    self.logger.error(
                        f"Error reading or validating imported config file: {e}. Proceeding with caution."
                    )
                    if not merge:
                        self.logger.warning(
                            "Cannot validate imported config, potential issues if dimensions mismatch."
                        )

            elif not merge:
                self.logger.warning(
                    "Imported archive does not contain a config file. Current config will be kept, ensure it's correct."
                )
                config_needs_saving = True  # Save current config after clearing

            # --- Clear existing data if not merging ---
            if not merge:
                self.logger.info("Merge=False. Clearing existing database...")
                for dir_name in ["vectors", "metadata", "raw_data", "index"]:
                    dir_path = os.path.join(self.storage_path, dir_name)
                    if os.path.isdir(dir_path):
                        try:
                            # More robust clearing: remove dir and recreate
                            shutil.rmtree(dir_path)
                            os.makedirs(dir_path)  # Recreate empty dir
                            self.logger.debug(
                                f"Cleared and recreated directory: {dir_path}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed to clear directory {dir_path}: {e}"
                            )
                            raise IOError(
                                f"Failed to clear existing database directory {dir_name}"
                            ) from e
                # Also remove old config if not merging, it will be replaced/regenerated
                if os.path.exists(self.config_path):
                    try:
                        os.remove(self.config_path)
                        self.logger.debug(
                            f"Removed old config file: {self.config_path}"
                        )
                    except OSError as e:
                        self.logger.error(
                            f"Could not remove old config file {self.config_path}: {e}"
                        )

            # --- Copy imported data ---
            imported_count = 0
            copied_files = 0
            for dir_name in ["vectors", "metadata", "raw_data", "index"]:
                src_dir = os.path.join(temp_dir, dir_name)
                dest_dir = os.path.join(self.storage_path, dir_name)
                os.makedirs(
                    dest_dir, exist_ok=True
                )  # Ensure destination exists

                if os.path.isdir(src_dir):
                    self.logger.debug(
                        f"Copying files from {src_dir} to {dest_dir}..."
                    )
                    # Use copytree for efficiency, handling overwrites if merge=True
                    try:
                        shutil.copytree(
                            src_dir,
                            dest_dir,
                            dirs_exist_ok=True,
                            copy_function=shutil.copy2,
                        )
                        # Count files copied in this step for logging
                        current_copied = sum(
                            len(files) for _, _, files in os.walk(src_dir)
                        )
                        copied_files += current_copied
                        if dir_name == "index":
                            # Count based on copied index files specifically
                            imported_count = len(
                                [
                                    f
                                    for f in os.listdir(dest_dir)
                                    if f.endswith(self.INDEX_MARKER_EXT)
                                ]
                            )
                    except Exception as copy_e:
                        self.logger.error(
                            f"Error copying directory {src_dir} to {dest_dir}: {copy_e}"
                        )
                        raise IOError(
                            f"Failed during data copy for directory {dir_name}"
                        ) from copy_e
                else:
                    self.logger.warning(
                        f"Source directory '{dir_name}' not found in extracted archive at {src_dir}."
                    )

            # --- Finalize Config ---
            # Save config if needed (either adopted from import or current one after clearing)
            if config_needs_saving:
                self._save_config()

            final_count = self.get_count()
            self.logger.info(
                f"Database import completed. Copied ~{copied_files} files. Final sample count: {final_count} (approx. {imported_count} indexed)."
            )
            return final_count


# Example Usage (Optional)
if __name__ == "__main__":

    # Make sure the embedding model server is running
    # CUDA_VISIBLE_DEVICES=1; VLLM_USE_V1=0; python -m vllm.entrypoints.openai.api_server --model sentence-transformers/all-MiniLM-L6-v2 --served-model-name all-MiniLM-L6-v2 --task embedding --port 8010

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger_main = logging.getLogger("DB_Example")

    # --- Configuration ---
    DB_PATH = "./sample_vector_db/main_example_" + datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    # Ensure the embedding model server is running at this URL if using vLLM
    EMBEDDING_URL = "http://localhost:8010/v1"  # Replace if needed
    # Use a model name compatible with your vLLM server or a standard OpenAI model if using their API directly
    MODEL_NAME = (
        "sentence-transformers/all-MiniLM-L6-v2"  # Example if vLLM serves this
    )
    # MODEL_NAME = (
    #     "intfloat/e5-mistral-7b-instruct"  # Example if vLLM serves this
    # )

    logger_main.info(f"Using database path: {DB_PATH}")
    logger_main.info(f"Using embedding model URL: {EMBEDDING_URL}")
    logger_main.info(f"Using embedding model name: {MODEL_NAME}")

    # --- Initialization ---
    try:
        db = SimpleVectorDB(
            storage_path=DB_PATH,
            embedding_model_name=MODEL_NAME,
            embedding_vllm_url=EMBEDDING_URL,
            task_representation_vector_db="metadata",  # Or "content"
            # Dimension will be auto-detected
        )
        logger_main.info(
            f"Database initialized. Dimension: {db.dimension}. Initial count: {db.get_count()}"
        )
    except Exception as e:
        logger_main.error(f"Failed to initialize database: {e}", exc_info=True)
        exit(1)

    # --- Add Samples ---
    sample_text = """
    This is the first sample document. It talks about apples and oranges.
    Fruits are healthy and delicious.
    """
    sample_metadata = {
        "topic": "fruits",
        "sentiment": "positive",
        "source": "manual",
        "length": len(sample_text),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        sample_id_1 = db.add_sample(sample_text, sample_metadata)
        logger_main.info(
            f"Added sample 1 with ID: {sample_id_1}. Count: {db.get_count()}"
        )
    except Exception as e:
        logger_main.error(f"Failed to add sample 1: {e}")

    another_sample = """
    A completely different topic: software engineering.
    Discussing algorithms, data structures, and system design.
    Python is a popular programming language.
    """
    another_metadata = {
        "topic": "software",
        "keywords": ["python", "algorithms", "system design"],
        "source": "manual",
        "length": len(another_sample),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        # Using a custom ID that might need sanitization for the index marker path
        custom_id = "tech/doc/001"
        sample_id_2 = db.add_sample(
            another_sample, another_metadata, custom_id=custom_id
        )
        logger_main.info(
            f"Added sample 2 with custom ID '{custom_id}' (stored as {sample_id_2}). Count: {db.get_count()}"
        )
    except Exception as e:
        logger_main.error(f"Failed to add sample 2: {e}")

    # --- Retrieve Sample ---
    retrieved = db.get_sample(sample_id_1)
    if retrieved:
        logger_main.info(
            f"\nRetrieved sample 1: ID={retrieved['id']}, Topic={retrieved['metadata'].get('topic')}"
        )
        logger_main.info(f"  Content: {retrieved['content'][:60]}...")
        logger_main.info(f"  Embedding shape: {retrieved['embedding'].shape}")
        logger_main.info(
            f"  Timestamp from marker: {retrieved.get('timestamp')}"
        )
    else:
        logger_main.warning(f"Could not retrieve sample {sample_id_1}")

    # --- Find Similar ---
    query_metadata = {"topic": "food", "keywords": ["apples", "healthy"]}
    query_text_for_meta = "\n".join(
        f"{k}: {v}" for k, v in sorted(query_metadata.items())
    )

    logger_main.info(f"\nFinding samples similar to metadata: {query_metadata}")
    try:
        # Use query_text if embedding based on metadata, or provide precomputed embedding
        similar_samples = db.find_similar(
            query=query_text_for_meta, metadata=query_metadata, top_n=3
        )
        logger_main.info("Similar samples found:")
        if similar_samples:
            for item in similar_samples:
                sample_id = item["sample_id"]
                similarity = item["similarity"]
                metadata = item["metadata"]
                content = item["content"]
                logger_main.info(
                    f"  - ID: {sample_id}, Score: {similarity:.4f}, Topic: {metadata.get('topic')}"
                )
                logger_main.info(f"  - Content: {content[:60]}...")
        else:
            logger_main.info("  No similar samples found.")
    except Exception as e:
        logger_main.error(f"Error during similarity search: {e}")

    # --- Update Metadata ---
    update_success = db.update_sample_metadata(
        sample_id_1, {"sentiment": "very positive", "reviewed": True}
    )
    logger_main.info(
        f"\nMetadata update for {sample_id_1} successful: {update_success}"
    )
    retrieved_updated = db.get_sample(sample_id_1)
    if retrieved_updated:
        logger_main.info(f"Updated metadata: {retrieved_updated['metadata']}")

    # --- Delete Sample ---
    delete_success = db.delete_sample(
        sample_id_2
    )  # Use the returned ID from add_sample
    logger_main.info(
        f"\nDeletion of {sample_id_2} successful: {delete_success}. Count: {db.get_count()}"
    )
    not_found = db.get_sample(sample_id_2)
    logger_main.info(
        f"Sample {sample_id_2} found after deletion: {not_found is not None}"
    )

    # --- Export ---
    EXPORT_PATH = (
        "./exported_db_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip"
    )
    exported_file = db.export_database(EXPORT_PATH)
    if exported_file:
        logger_main.info(f"\nDatabase exported to: {exported_file}")

        # --- Import (Example: into a new DB instance, clearing first) ---
        IMPORT_DB_PATH = DB_PATH + "/imported_vector_db"
        logger_main.info(
            f"\nAttempting to import into new DB at: {IMPORT_DB_PATH}"
        )
        try:
            # Initialize with the same model details expected from the export
            import_db = SimpleVectorDB(
                storage_path=IMPORT_DB_PATH,
                embedding_model_name=MODEL_NAME,  # Should match exported DB if not merging
                embedding_vllm_url=EMBEDDING_URL,
                task_representation_vector_db="metadata",  # Should match exported DB
                # Dimension will be loaded from imported config if present
            )
            imported_count = import_db.import_database(
                exported_file, merge=False
            )
            logger_main.info(
                f"Import successful. New DB count: {imported_count} (should be {db.get_count()})"
            )
            # Verify content
            imported_ids = import_db.get_all_sample_ids()
            logger_main.info(f"IDs in imported DB: {imported_ids}")
            if sample_id_1 in imported_ids:
                logger_main.info(f"Sample {sample_id_1} successfully imported.")
            else:
                logger_main.warning(
                    f"Sample {sample_id_1} missing after import."
                )
            if sample_id_2 in imported_ids:
                logger_main.warning(
                    f"Sample {sample_id_2} present after import (should have been deleted before export)."
                )

        except Exception as e:
            logger_main.error(f"Error during import test: {e}", exc_info=True)

    else:
        logger_main.error("Database export failed.")
