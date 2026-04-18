"""
Main script to build synthetic task vector database.

This script loads synthetically generated tasks from an experiment directory,
extracts their example instructions, and stores them in a vector database
for semantic similarity search.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark_question_db.simple_vectordb import SimpleVectorDB


class MockVectorDB(SimpleVectorDB):
    """
    Mock version of SimpleVectorDB that generates random embeddings.
    Useful for development and testing without requiring an embedding server.
    """

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a random embedding vector."""
        # Use hash of text as seed for reproducibility
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return np.random.randn(self.dimension).astype(np.float32)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SyntheticTaskVectorDB")


class SyntheticTask:
    """
    Represents a synthetic task with its content and metadata.
    """

    def __init__(
        self,
        task_id: str,
        example_instruction: str,
        name_of_task: str,
        description_of_task: str,
        capability_being_measured: str,
        estimated_human_difficulty: str,
    ):
        self.task_id = task_id
        self.example_instruction = example_instruction
        self.metadata = {
            "task_id": task_id,
            "name_of_task": name_of_task,
            "description_of_task": description_of_task,
            "capability_being_measured": capability_being_measured,
            "estimated_human_difficulty": estimated_human_difficulty,
        }

    @classmethod
    def from_json(cls, task_dir: Path) -> Optional["SyntheticTask"]:
        """
        Load a synthetic task from its directory.

        Args:
            task_dir: Path to the task directory

        Returns:
            SyntheticTask instance or None if loading fails
        """
        task_json_path = task_dir / "task.json"

        if not task_json_path.exists():
            return None

        try:
            with open(task_json_path, "r") as f:
                data = json.load(f)

            # Extract required field
            example_instruction = data.get("example_instruction")
            if not example_instruction:
                return None

            # Extract metadata fields (with defaults for missing fields)
            task_id = task_dir.name
            name_of_task = data.get("name_of_task", "")
            description_of_task = data.get("description_of_task", "")
            capability_being_measured = data.get("capability_being_measured", "")
            estimated_human_difficulty = data.get("estimated_human_difficulty", "")

            return cls(
                task_id=task_id,
                example_instruction=example_instruction,
                name_of_task=name_of_task,
                description_of_task=description_of_task,
                capability_being_measured=capability_being_measured,
                estimated_human_difficulty=estimated_human_difficulty,
            )

        except (json.JSONDecodeError, IOError) as e:
            return None


def discover_tasks(
    experiment_dir: str,
    logger: logging.Logger,
) -> List[Path]:
    """
    Discover all task directories in the experiment's pool.

    Args:
        experiment_dir: Path to experiment directory
        logger: Logger instance

    Returns:
        List of task directory paths
    """
    pool_path = Path(experiment_dir) / "generated_tasks" / "pool"

    if not pool_path.exists():
        logger.error(f"Pool directory not found: {pool_path}")
        return []

    # Find all subdirectories in the pool
    task_dirs = [d for d in pool_path.iterdir() if d.is_dir()]
    logger.info(f"Discovered {len(task_dirs)} task directories in {pool_path}")

    return task_dirs


def load_tasks(
    task_dirs: List[Path],
    logger: logging.Logger,
) -> List[SyntheticTask]:
    """
    Load synthetic tasks from their directories.

    Args:
        task_dirs: List of task directory paths
        logger: Logger instance

    Returns:
        List of successfully loaded SyntheticTask objects
    """
    tasks = []
    failed_count = 0

    for task_dir in task_dirs:
        task = SyntheticTask.from_json(task_dir)
        if task:
            tasks.append(task)
        else:
            failed_count += 1
            logger.debug(f"Failed to load task from {task_dir.name}")

    logger.info(f"Loaded {len(tasks)} tasks successfully")
    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} tasks")

    return tasks


def add_tasks_to_db(
    db: SimpleVectorDB,
    tasks: List[SyntheticTask],
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Add tasks to the vector database.

    Args:
        db: Vector database instance
        tasks: List of tasks to add
        logger: Logger instance

    Returns:
        Dictionary with statistics (added, failed)
    """
    stats = {"added": 0, "failed": 0}

    for task in tasks:
        try:
            db.add_sample(
                content=task.example_instruction,
                metadata=task.metadata,
                custom_id=task.task_id,
            )
            stats["added"] += 1

            if stats["added"] % 100 == 0:
                logger.info(f"  Progress: {stats['added']} tasks added...")

        except Exception as e:
            logger.error(f"Failed to add task {task.task_id}: {e}")
            stats["failed"] += 1

    return stats


def build_database(
    experiment_dir: str,
    storage_path: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_vllm_url: str = "http://localhost:8010/v1",
    mock_embeddings: bool = False,
    verbose: bool = False,
) -> None:
    """
    Build the synthetic task vector database.

    Args:
        experiment_dir: Path to experiment directory
        storage_path: Path to store the database
        embedding_model_name: Name of embedding model
        embedding_vllm_url: URL of embedding server
        mock_embeddings: Use mock embeddings instead of real server
        verbose: Enable verbose logging
    """
    logger = setup_logging(verbose)

    logger.info("=" * 80)
    logger.info("Building Synthetic Task Vector Database")
    logger.info("=" * 80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Storage path: {storage_path}")
    logger.info(f"Embedding model: {embedding_model_name}")
    if mock_embeddings:
        logger.info("Mode: MOCK EMBEDDINGS (for development)")
    else:
        logger.info(f"Embedding server: {embedding_vllm_url}")
    logger.info("=" * 80)

    # Discover tasks
    logger.info("\nDiscovering tasks...")
    task_dirs = discover_tasks(experiment_dir, logger)

    if not task_dirs:
        logger.error("No task directories found. Exiting.")
        sys.exit(1)

    # Load tasks
    logger.info("\nLoading tasks...")
    tasks = load_tasks(task_dirs, logger)

    if not tasks:
        logger.error("No tasks loaded successfully. Exiting.")
        sys.exit(1)

    # Initialize vector database
    logger.info("\nInitializing vector database...")
    try:
        if mock_embeddings:
            # Use mock database with fixed dimension
            db = MockVectorDB(
                storage_path=storage_path,
                embedding_model_name=embedding_model_name,
                embedding_vllm_url=embedding_vllm_url,
                dimension=384,  # Standard dimension for all-MiniLM-L6-v2
                task_representation_vector_db="content",
            )
        else:
            db = SimpleVectorDB(
                storage_path=storage_path,
                embedding_model_name=embedding_model_name,
                embedding_vllm_url=embedding_vllm_url,
                task_representation_vector_db="content",
            )
        logger.info(f"  ✓ Database initialized (dimension: {db.dimension})")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        sys.exit(1)

    # Add tasks to database
    logger.info("\nAdding tasks to database...")
    stats = add_tasks_to_db(db, tasks, logger)

    # Print final statistics
    logger.info("\n" + "=" * 80)
    logger.info("Database Build Complete")
    logger.info("=" * 80)
    logger.info(f"Total tasks added: {stats['added']}")
    logger.info(f"Total failures: {stats['failed']}")
    logger.info(f"Final database count: {db.get_count()}")
    logger.info(f"\nDatabase saved to: {storage_path}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build synthetic task vector database from experiment outputs"
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="outputs/qwen2.5",
        help="Path to experiment directory (default: outputs/qwen2.5)",
    )

    parser.add_argument(
        "--storage-path",
        type=str,
        default="qwen2.5_synth_task-vectordb",
        help="Path to store the database (default: qwen2.5_synth_task-vectordb)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model",
    )

    parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8010/v1",
        help="URL of the embedding server",
    )

    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use mock embeddings instead of real server (for development)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

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
