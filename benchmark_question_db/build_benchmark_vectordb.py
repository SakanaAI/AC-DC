"""
Main script to build benchmark question vector database.

This script loads questions from various AI benchmarks, formats them,
and stores them in a vector database for semantic similarity search.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np

from benchmark_question_db.simple_vectordb import SimpleVectorDB
from benchmark_question_db.loaders import (
    BenchmarkSample,
    MMLULoader,
    MMLUProLoader,
    GSM8KLoader,
    HumanEvalLoader,
    MBPPLoader,
    BBHLoader,
    MATHLoader,
    GPQALoader,
)


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
    return logging.getLogger("BenchmarkVectorDB")


def load_benchmark_samples(
    benchmark_name: str,
    loader_class,
    n_per_subgroup: int,
    total_samples: int,
    random_seed: int,
    include_mcq_options: bool,
    logger: logging.Logger,
) -> List[BenchmarkSample]:
    """
    Load samples from a benchmark using its loader.

    Args:
        benchmark_name: Name of the benchmark
        loader_class: Loader class to instantiate
        n_per_subgroup: Samples per subgroup
        total_samples: Total samples if no subgroups
        random_seed: Random seed
        include_mcq_options: Include MCQ options in formatting
        logger: Logger instance

    Returns:
        List of BenchmarkSample objects
    """
    logger.info(f"Loading {benchmark_name}...")
    try:
        loader = loader_class(random_seed=random_seed, include_mcq_options=include_mcq_options)
        samples = loader.sample_questions(
            n_per_subgroup=n_per_subgroup,
            total_samples=total_samples,
        )
        logger.info(
            f"  ✓ Loaded {len(samples)} samples from {benchmark_name}"
        )
        return samples
    except Exception as e:
        logger.error(f"  ✗ Failed to load {benchmark_name}: {e}", exc_info=True)
        return []


def add_samples_to_db(
    db: SimpleVectorDB,
    samples: List[BenchmarkSample],
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Add samples to the vector database.

    Args:
        db: Vector database instance
        samples: List of samples to add
        logger: Logger instance

    Returns:
        Dictionary with statistics (added, failed)
    """
    stats = {"added": 0, "failed": 0}

    for sample in samples:
        try:
            db.add_sample(
                content=sample.content,
                metadata=sample.metadata,
                custom_id=sample.question_id,
            )
            stats["added"] += 1
        except Exception as e:
            logger.error(
                f"Failed to add sample {sample.question_id}: {e}"
            )
            stats["failed"] += 1

    return stats


def build_database(
    storage_path: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_vllm_url: str = "http://localhost:8010/v1",
    n_per_subgroup: int = 10,
    total_samples: int = 30,
    random_seed: int = 42,
    mock_embeddings: bool = False,
    include_mcq_options: bool = False,
    verbose: bool = False,
) -> None:
    """
    Build the benchmark question vector database.

    Args:
        storage_path: Path to store the database
        embedding_model_name: Name of embedding model
        embedding_vllm_url: URL of embedding server
        n_per_subgroup: Samples per subgroup
        total_samples: Total samples for benchmarks without subgroups
        random_seed: Random seed for reproducibility
        mock_embeddings: Use mock embeddings instead of real server
        include_mcq_options: Include multiple-choice options in question formatting
        verbose: Enable verbose logging
    """
    logger = setup_logging(verbose)

    logger.info("=" * 80)
    logger.info("Building Benchmark Question Vector Database")
    logger.info("=" * 80)
    logger.info(f"Storage path: {storage_path}")
    logger.info(f"Embedding model: {embedding_model_name}")
    if mock_embeddings:
        logger.info("Mode: MOCK EMBEDDINGS (for development)")
    else:
        logger.info(f"Embedding server: {embedding_vllm_url}")
    logger.info(f"Samples per subgroup: {n_per_subgroup}")
    logger.info(f"Total samples (no subgroups): {total_samples}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Include MCQ options: {include_mcq_options}")
    logger.info("=" * 80)

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

    # Define benchmarks to load
    benchmarks = [
        ("MMLU", MMLULoader),
        ("MMLU Pro", MMLUProLoader),
        ("Big Bench Hard", BBHLoader),
        ("MATH", MATHLoader),
        ("GSM8K", GSM8KLoader),
        ("HumanEval", HumanEvalLoader),
        ("MBPP", MBPPLoader),
        ("GPQA", GPQALoader),
    ]

    # Load and add samples from each benchmark
    logger.info("\nLoading benchmarks...")
    logger.info("=" * 80)

    total_stats = {"added": 0, "failed": 0}
    benchmark_stats = {}

    for benchmark_name, loader_class in benchmarks:
        samples = load_benchmark_samples(
            benchmark_name,
            loader_class,
            n_per_subgroup,
            total_samples,
            random_seed,
            include_mcq_options,
            logger,
        )

        if samples:
            logger.info(f"Adding {len(samples)} samples to database...")
            stats = add_samples_to_db(db, samples, logger)
            total_stats["added"] += stats["added"]
            total_stats["failed"] += stats["failed"]
            benchmark_stats[benchmark_name] = stats
            logger.info(
                f"  ✓ Added {stats['added']} samples (failed: {stats['failed']})"
            )

    # Print final statistics
    logger.info("\n" + "=" * 80)
    logger.info("Database Build Complete")
    logger.info("=" * 80)
    logger.info(f"Total samples added: {total_stats['added']}")
    logger.info(f"Total failures: {total_stats['failed']}")
    logger.info(f"Final database count: {db.get_count()}")
    logger.info("\nBreakdown by benchmark:")
    for benchmark_name, stats in benchmark_stats.items():
        logger.info(f"  {benchmark_name}: {stats['added']} samples")

    logger.info(f"\nDatabase saved to: {storage_path}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build benchmark question vector database"
    )

    parser.add_argument(
        "--storage-path",
        type=str,
        default="./benchmark_vector_db",
        help="Path to store the database",
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
        "--n-per-subgroup",
        type=int,
        default=10,
        help="Number of samples per subgroup (for benchmarks with categories)",
    )

    parser.add_argument(
        "--total-samples",
        type=int,
        default=30,
        help="Total samples for benchmarks without subgroups",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use mock embeddings instead of real server (for development)",
    )

    parser.add_argument(
        "--include-mcq-options",
        action="store_true",
        help="Include multiple-choice options in question formatting (default: False)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    build_database(
        storage_path=args.storage_path,
        embedding_model_name=args.embedding_model,
        embedding_vllm_url=args.embedding_url,
        n_per_subgroup=args.n_per_subgroup,
        total_samples=args.total_samples,
        random_seed=args.random_seed,
        mock_embeddings=args.mock_embeddings,
        include_mcq_options=args.include_mcq_options,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
