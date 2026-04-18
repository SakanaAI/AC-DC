"""
Build a vector database from synthetic tasks.

This script scans an experiment output directory for synthetic tasks and creates
a vector database where each task's example_instruction is embedded.

Usage:
    # With embedding server
    python build_synthetic_task_vectordb.py \
        --experiment-dir outputs/qwen2.5 \
        --storage-path ./synthetic_task_vectordb \
        --embedding-model intfloat/e5-mistral-7b-instruct \
        --embedding-url http://localhost:8010/v1

    # With mock embeddings (for testing)
    python build_synthetic_task_vectordb.py \
        --experiment-dir outputs/qwen2.5 \
        --storage-path ./test_synthetic_task_vectordb \
        --mock-embeddings \
        --verbose
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add parent directory to path to import simple_vectordb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_question_db.simple_vectordb import SimpleVectorDB


class MockVectorDB(SimpleVectorDB):
    """Mock version that generates reproducible random embeddings for testing."""

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a reproducible random embedding based on text hash."""
        # Use hash of text as seed for reproducibility
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        embedding = np.random.randn(self.dimension).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


def load_task_from_directory(task_dir: Path) -> Dict[str, Any]:
    """
    Load a task from its directory.

    Args:
        task_dir: Path to task directory (e.g., task_681_conditional_probability_and_total_probability)

    Returns:
        Dictionary containing:
            - task_id: The ID from directory name (e.g., "task_681_conditional_probability_and_total_probability")
            - example_instruction: The instruction text to embed
            - metadata: Other task information
    """
    task_json_path = task_dir / "task.json"

    if not task_json_path.exists():
        raise FileNotFoundError(f"task.json not found in {task_dir}")

    with open(task_json_path, 'r') as f:
        task_data = json.load(f)

    # Extract task ID from directory name
    task_id = task_dir.name

    # Get the example instruction (this is what we'll embed)
    example_instruction = task_data.get("example_instruction", "")

    if not example_instruction:
        raise ValueError(f"No example_instruction found in {task_json_path}")

    # Prepare metadata (everything except example_instruction)
    metadata = {
        "task_id": task_id,
        "name_of_task": task_data.get("name_of_task", ""),
        "description_of_task": task_data.get("description_of_task", ""),
        "capability_being_measured": task_data.get("capability_being_measured", ""),
        "estimated_human_difficulty": task_data.get("estimated_human_difficulty", ""),
    }

    return {
        "task_id": task_id,
        "example_instruction": example_instruction,
        "metadata": metadata,
        "raw_data": task_data,
    }


def find_all_tasks(experiment_dir: Path) -> List[Path]:
    """
    Find all task directories in the experiment output directory.

    Args:
        experiment_dir: Path to experiment directory (e.g., outputs/qwen2.5)

    Returns:
        List of paths to task directories
    """
    pool_dir = experiment_dir / "generated_tasks" / "pool"

    if not pool_dir.exists():
        raise FileNotFoundError(
            f"Pool directory not found: {pool_dir}\n"
            f"Expected structure: {experiment_dir}/generated_tasks/pool/task_*"
        )

    # Find all directories matching task_* pattern
    task_dirs = sorted([
        d for d in pool_dir.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    ])

    return task_dirs


def build_database(
    experiment_dir: str,
    storage_path: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_vllm_url: str = "http://localhost:8010/v1",
    mock_embeddings: bool = False,
    verbose: bool = False,
) -> None:
    """
    Build a vector database from synthetic tasks.

    Args:
        experiment_dir: Path to experiment directory (e.g., outputs/qwen2.5)
        storage_path: Where to store the vector database
        embedding_model_name: Name of the embedding model
        embedding_vllm_url: URL of the embedding server (OpenAI-compatible)
        mock_embeddings: If True, use mock random embeddings (for testing)
        verbose: If True, print detailed progress
    """
    # Set up logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    experiment_path = Path(experiment_dir)

    # Find all task directories
    logger.info(f"Scanning for tasks in {experiment_path}")
    task_dirs = find_all_tasks(experiment_path)
    logger.info(f"Found {len(task_dirs)} task directories")

    if len(task_dirs) == 0:
        logger.warning("No tasks found!")
        return

    # Initialize vector database
    if mock_embeddings:
        logger.info("Using mock embeddings (reproducible random vectors)")
        # For mock embeddings, use a default dimension of 384
        db = MockVectorDB(
            storage_path=storage_path,
            embedding_model_name=embedding_model_name,
            embedding_vllm_url=None,
            task_representation_vector_db="content",
            dimension=384,  # Default dimension for mock embeddings
        )
    else:
        logger.info(f"Using embedding model: {embedding_model_name}")
        logger.info(f"Embedding server URL: {embedding_vllm_url}")
        db = SimpleVectorDB(
            storage_path=storage_path,
            embedding_model_name=embedding_model_name,
            embedding_vllm_url=embedding_vllm_url,
            task_representation_vector_db="content",
        )

    # Process each task
    success_count = 0
    error_count = 0

    print(f"\nProcessing {len(task_dirs)} tasks...")
    print("=" * 80)

    for i, task_dir in enumerate(task_dirs, 1):
        try:
            # Load task data
            task = load_task_from_directory(task_dir)

            if verbose:
                logger.info(f"[{i}/{len(task_dirs)}] Processing {task['task_id']}")
            else:
                # Progress indicator
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(task_dirs)} ({100*i//len(task_dirs)}%)")

            # Add to database
            # The content to embed is the example_instruction
            # The custom_id is the task_id (e.g., task_681_conditional_probability_and_total_probability)
            db.add_sample(
                content=task['example_instruction'],
                metadata=task['metadata'],
                custom_id=task['task_id'],
            )

            success_count += 1

            if verbose:
                logger.info(f"  ✓ Added to database")
                logger.info(f"  Instruction preview: {task['example_instruction'][:100]}...")

        except Exception as e:
            logger.error(f"Error processing {task_dir.name}: {e}")
            error_count += 1
            continue

    print("=" * 80)
    print(f"\nDatabase build complete!")
    print(f"  Total tasks processed: {len(task_dirs)}")
    print(f"  Successfully added: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Storage path: {storage_path}")

    # Show some statistics
    if success_count > 0:
        print(f"\nDatabase Statistics:")
        print(f"  Embedding model: {embedding_model_name}")
        print(f"  Embedding dimension: {db.dimension}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a vector database from synthetic tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with embedding server
  python build_synthetic_task_vectordb.py \\
      --experiment-dir outputs/qwen2.5 \\
      --storage-path ./synthetic_task_vectordb \\
      --embedding-model intfloat/e5-mistral-7b-instruct \\
      --embedding-url http://localhost:8010/v1

  # Build with mock embeddings (testing)
  python build_synthetic_task_vectordb.py \\
      --experiment-dir outputs/qwen2.5 \\
      --storage-path ./test_synthetic_task_vectordb \\
      --mock-embeddings \\
      --verbose
        """
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment output directory (e.g., outputs/qwen2.5)"
    )

    parser.add_argument(
        "--storage-path",
        type=str,
        required=True,
        help="Path where the vector database will be stored"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model (default: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8010/v1",
        help="URL of the OpenAI-compatible embedding server (default: http://localhost:8010/v1)"
    )

    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use mock random embeddings instead of real embeddings (for testing)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()

    # Build the database
    build_database(
        experiment_dir=args.experiment_dir,
        storage_path=args.storage_path,
        embedding_model_name=args.embedding_model,
        embedding_vllm_url=args.embedding_url,
        mock_embeddings=args.mock_embeddings,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
