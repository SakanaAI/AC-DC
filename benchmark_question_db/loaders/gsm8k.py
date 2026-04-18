"""
GSM8K benchmark loader.
"""

from typing import Any, Dict, List, Optional
from datasets import Dataset, load_dataset
import random

from benchmark_question_db.loaders.base import BenchmarkLoader, BenchmarkSample


class GSM8KLoader(BenchmarkLoader):
    """Loader for GSM8K (Grade School Math 8K) dataset."""

    def __init__(self, random_seed: int = 42, include_mcq_options: bool = False):
        super().__init__("GSM8K", random_seed, include_mcq_options)
        self._dataset = None

    def load_dataset(self) -> Dataset:
        """Load GSM8K dataset from HuggingFace."""
        if self._dataset is None:
            self._dataset = load_dataset("openai/gsm8k", "main", split="test")
        return self._dataset

    def get_subgroups(self) -> Optional[List[str]]:
        """GSM8K has no subgroups."""
        return None

    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format GSM8K question as a math word problem.

        Just return the question text without the answer.
        """
        question = item["question"]
        return f"{question}"

    def extract_metadata(
        self, item: Dict[str, Any], subgroup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata including answer."""
        metadata = super().extract_metadata(item, subgroup)

        # GSM8K answers include step-by-step solutions
        # Extract the final answer (after ####)
        answer = item["answer"]
        if "####" in answer:
            final_answer = answer.split("####")[-1].strip()
        else:
            final_answer = answer

        metadata["answer"] = final_answer
        metadata["full_solution"] = answer
        metadata["question_type"] = "math_word_problem"

        return metadata

    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """Not used for GSM8K (no subgroups)."""
        raise NotImplementedError("GSM8K does not have subgroups")

    def _sample_from_dataset(self, dataset: Dataset, n: int) -> List[BenchmarkSample]:
        """Sample n questions from GSM8K."""
        total_available = len(dataset)
        if total_available < n:
            print(f"Warning: GSM8K has only {total_available} samples, requested {n}")
            indices = list(range(total_available))
        else:
            indices = random.sample(range(total_available), n)

        samples = []
        for idx in indices:
            item = dataset[idx]
            sample = self._create_sample(item, idx, None)
            samples.append(sample)

        return samples
