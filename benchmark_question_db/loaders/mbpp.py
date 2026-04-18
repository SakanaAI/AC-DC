"""
MBPP benchmark loader.
"""

from typing import Any, Dict, List, Optional
from datasets import Dataset, load_dataset
import random

from benchmark_question_db.loaders.base import BenchmarkLoader, BenchmarkSample


class MBPPLoader(BenchmarkLoader):
    """Loader for MBPP (Mostly Basic Python Problems) benchmark."""

    def __init__(self, random_seed: int = 42, include_mcq_options: bool = False):
        super().__init__("MBPP", random_seed, include_mcq_options)
        self._dataset = None

    def load_dataset(self) -> Dataset:
        """Load MBPP dataset from HuggingFace."""
        if self._dataset is None:
            self._dataset = load_dataset("google-research-datasets/mbpp", split="test")
        return self._dataset

    def get_subgroups(self) -> Optional[List[str]]:
        """MBPP has no subgroups."""
        return None

    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format MBPP question as a code generation problem.

        Returns the problem description (text field).
        """
        text = item["text"]
        return f"{text}"

    def extract_metadata(
        self, item: Dict[str, Any], subgroup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata including task ID."""
        metadata = super().extract_metadata(item, subgroup)

        metadata["task_id"] = item["task_id"]
        metadata["question_type"] = "code_generation"
        metadata["language"] = "python"

        # Store solution and tests separately (not in embedding)
        if "code" in item:
            metadata["canonical_solution"] = item["code"]
        if "test_list" in item:
            metadata["test_cases"] = item["test_list"]

        return metadata

    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """Not used for MBPP (no subgroups)."""
        raise NotImplementedError("MBPP does not have subgroups")

    def _sample_from_dataset(self, dataset: Dataset, n: int) -> List[BenchmarkSample]:
        """Sample n questions from MBPP."""
        total_available = len(dataset)
        if total_available < n:
            print(f"Warning: MBPP has only {total_available} samples, requested {n}")
            indices = list(range(total_available))
        else:
            indices = random.sample(range(total_available), n)

        samples = []
        for idx in indices:
            item = dataset[idx]
            sample = self._create_sample(item, idx, None)
            samples.append(sample)

        return samples
