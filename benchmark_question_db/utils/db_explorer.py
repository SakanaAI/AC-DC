"""
Utility script to explore and query the benchmark question vector database.
"""

import argparse
import json
import logging
from collections import Counter
from typing import Dict, List, Optional

from benchmark_question_db.simple_vectordb import SimpleVectorDB


class BenchmarkDBExplorer:
    """Explorer for benchmark question vector database."""

    def __init__(self, db_path: str):
        """
        Initialize the explorer.

        Args:
            db_path: Path to the vector database
        """
        self.db = SimpleVectorDB(
            storage_path=db_path,
            task_representation_vector_db="content",
        )
        self.logger = logging.getLogger("DBExplorer")

    def print_statistics(self) -> Dict:
        """Print database statistics."""
        print("\n" + "=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)

        total_count = self.db.get_count()
        print(f"\nTotal samples: {total_count}")

        # Get all sample IDs and load metadata
        sample_ids = self.db.get_all_sample_ids()

        # Collect statistics
        benchmarks = []
        subgroups = []
        question_types = []

        for sample_id in sample_ids:
            sample = self.db.get_sample(sample_id)
            if sample:
                metadata = sample["metadata"]
                benchmarks.append(metadata.get("benchmark", "Unknown"))
                if "subgroup" in metadata:
                    subgroups.append(
                        f"{metadata['benchmark']}:{metadata['subgroup']}"
                    )
                question_types.append(
                    metadata.get("question_type", "Unknown")
                )

        # Print breakdowns
        print("\nBy Benchmark:")
        benchmark_counts = Counter(benchmarks)
        for benchmark, count in sorted(benchmark_counts.items()):
            print(f"  {benchmark}: {count}")

        print("\nBy Question Type:")
        type_counts = Counter(question_types)
        for qtype, count in sorted(type_counts.items()):
            print(f"  {qtype}: {count}")

        print(
            f"\nTotal unique subgroups: {len(set(subgroups))}"
        )

        stats = {
            "total": total_count,
            "benchmarks": dict(benchmark_counts),
            "question_types": dict(type_counts),
            "unique_subgroups": len(set(subgroups)),
        }

        return stats

    def list_samples(
        self,
        benchmark: Optional[str] = None,
        limit: int = 10,
        show_content: bool = False,
    ) -> None:
        """
        List samples in the database.

        Args:
            benchmark: Filter by benchmark name
            limit: Maximum samples to show
            show_content: Whether to show question content
        """
        print("\n" + "=" * 80)
        print("SAMPLE LISTING")
        print("=" * 80)

        sample_ids = self.db.get_all_sample_ids()
        count = 0

        for sample_id in sample_ids:
            if count >= limit:
                break

            sample = self.db.get_sample(sample_id)
            if not sample:
                continue

            metadata = sample["metadata"]

            # Filter by benchmark if specified
            if benchmark and metadata.get("benchmark") != benchmark:
                continue

            count += 1
            print(f"\n[{count}] ID: {sample_id}")
            print(f"    Benchmark: {metadata.get('benchmark')}")
            if "subgroup" in metadata:
                print(f"    Subgroup: {metadata['subgroup']}")
            print(f"    Type: {metadata.get('question_type')}")

            if show_content:
                content = sample["content"]
                # Truncate long content
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"\n    Content:\n    {content}\n")

        if count == 0:
            print("\nNo samples found.")
        else:
            print(f"\nShowing {count} of {len(sample_ids)} total samples")

    def search_similar(
        self,
        query: str,
        top_n: int = 5,
        benchmark: Optional[str] = None,
        show_content: bool = True,
    ) -> None:
        """
        Search for similar questions.

        Args:
            query: Query text
            top_n: Number of results to return
            benchmark: Filter by benchmark
            show_content: Whether to show question content
        """
        print("\n" + "=" * 80)
        print("SIMILARITY SEARCH")
        print("=" * 80)
        print(f"\nQuery: {query}")
        print(f"Top {top_n} results:")

        results = self.db.find_similar(
            query=query,
            top_n=top_n * 2 if benchmark else top_n,  # Get more if filtering
        )

        # Filter by benchmark if specified
        if benchmark:
            results = [
                r
                for r in results
                if r["metadata"].get("benchmark") == benchmark
            ]
            results = results[:top_n]

        if not results:
            print("\nNo results found.")
            return

        for i, result in enumerate(results):
            print(f"\n[{i+1}] ID: {result['sample_id']}")
            print(f"    Similarity: {result['similarity']:.4f}")
            print(f"    Benchmark: {result['metadata'].get('benchmark')}")
            if "subgroup" in result["metadata"]:
                print(f"    Subgroup: {result['metadata']['subgroup']}")
            print(f"    Type: {result['metadata'].get('question_type')}")

            if show_content:
                content = result["content"]
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"\n    Content:\n    {content}\n")

    def get_sample_by_id(self, sample_id: str) -> None:
        """
        Retrieve and display a specific sample.

        Args:
            sample_id: Sample ID to retrieve
        """
        print("\n" + "=" * 80)
        print("SAMPLE DETAILS")
        print("=" * 80)

        sample = self.db.get_sample(sample_id)

        if not sample:
            print(f"\nSample not found: {sample_id}")
            return

        print(f"\nID: {sample['id']}")
        print(f"\nMetadata:")
        print(json.dumps(sample["metadata"], indent=2))
        print(f"\nContent:")
        print(sample["content"])
        print(f"\nEmbedding shape: {sample['embedding'].shape}")


def main():
    """Main entry point for the explorer."""
    parser = argparse.ArgumentParser(
        description="Explore benchmark question vector database"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="./benchmark_vector_db",
        help="Path to the vector database",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # List command
    list_parser = subparsers.add_parser("list", help="List samples")
    list_parser.add_argument(
        "--benchmark", type=str, help="Filter by benchmark"
    )
    list_parser.add_argument(
        "--limit", type=int, default=10, help="Max samples to show"
    )
    list_parser.add_argument(
        "--show-content", action="store_true", help="Show question content"
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search for similar questions"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--top-n", type=int, default=5, help="Number of results"
    )
    search_parser.add_argument(
        "--benchmark", type=str, help="Filter by benchmark"
    )
    search_parser.add_argument(
        "--no-content",
        action="store_true",
        help="Don't show question content",
    )

    # Get command
    get_parser = subparsers.add_parser("get", help="Get sample by ID")
    get_parser.add_argument("sample_id", type=str, help="Sample ID")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize explorer
    explorer = BenchmarkDBExplorer(args.db_path)

    # Execute command
    if args.command == "stats":
        explorer.print_statistics()

    elif args.command == "list":
        explorer.list_samples(
            benchmark=args.benchmark,
            limit=args.limit,
            show_content=args.show_content,
        )

    elif args.command == "search":
        explorer.search_similar(
            query=args.query,
            top_n=args.top_n,
            benchmark=args.benchmark,
            show_content=not args.no_content,
        )

    elif args.command == "get":
        explorer.get_sample_by_id(args.sample_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
