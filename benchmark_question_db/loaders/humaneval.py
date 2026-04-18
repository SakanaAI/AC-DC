"""
HumanEval benchmark loader.
"""

from typing import Any, Dict, List, Optional
from datasets import Dataset, load_dataset
import random

from benchmark_question_db.loaders.base import BenchmarkLoader, BenchmarkSample


class HumanEvalLoader(BenchmarkLoader):
    """Loader for HumanEval code generation benchmark."""

    def __init__(self, random_seed: int = 42, include_mcq_options: bool = False):
        super().__init__("HumanEval", random_seed, include_mcq_options)
        self._dataset = None

    def load_dataset(self) -> Dataset:
        """Load HumanEval dataset from HuggingFace."""
        if self._dataset is None:
            self._dataset = load_dataset("openai/openai_humaneval", split="test")
        return self._dataset

    def get_subgroups(self) -> Optional[List[str]]:
        """HumanEval has no subgroups."""
        return None

    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format HumanEval question as a code generation problem.

        Returns the prompt which includes the function signature and docstring.
        """
        prompt = item["prompt"]
        return f"{prompt.strip()}"

    def extract_metadata(
        self, item: Dict[str, Any], subgroup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata including task ID and entry point."""
        metadata = super().extract_metadata(item, subgroup)

        metadata["task_id"] = item["task_id"]
        metadata["entry_point"] = item["entry_point"]
        metadata["question_type"] = "code_generation"
        metadata["language"] = "python"

        # Store solution separately (not in embedding)
        if "canonical_solution" in item:
            metadata["canonical_solution"] = item["canonical_solution"]

        return metadata

    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """Not used for HumanEval (no subgroups)."""
        raise NotImplementedError("HumanEval does not have subgroups")

    def _sample_from_dataset(self, dataset: Dataset, n: int) -> List[BenchmarkSample]:
        """Sample n questions from HumanEval."""
        total_available = len(dataset)
        if total_available < n:
            print(
                f"Warning: HumanEval has only {total_available} samples, requested {n}"
            )
            indices = list(range(total_available))
        else:
            indices = random.sample(range(total_available), n)

        samples = []
        for idx in indices:
            item = dataset[idx]
            sample = self._create_sample(item, idx, None)
            samples.append(sample)

        return samples
