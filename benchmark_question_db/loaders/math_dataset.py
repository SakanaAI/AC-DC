"""
MATH dataset loader (Minerva Math).
"""

from typing import Any, Dict, List, Optional
from datasets import Dataset, load_dataset, get_dataset_config_names
import random

from benchmark_question_db.loaders.base import BenchmarkLoader, BenchmarkSample


class MATHLoader(BenchmarkLoader):
    """Loader for MATH (Mathematics Aptitude Test of Heuristics) dataset."""

    def __init__(self, random_seed: int = 42, include_mcq_options: bool = False):
        super().__init__("MATH", random_seed, include_mcq_options)
        self._datasets = {}  # Cache for each category's dataset
        self._subgroups = None

    def load_dataset(self) -> Optional[Dataset]:
        """
        MATH has multiple configs (categories), so we load them separately.
        This method returns None; use load_category_dataset instead.
        """
        return None

    def load_category_dataset(self, category: str) -> Dataset:
        """Load dataset for a specific MATH category."""
        if category not in self._datasets:
            self._datasets[category] = load_dataset(
                "EleutherAI/hendrycks_math", category, split="test"
            )
        return self._datasets[category]

    def get_subgroups(self) -> Optional[List[str]]:
        """Get all categories in MATH."""
        if self._subgroups is None:
            self._subgroups = sorted(
                get_dataset_config_names("EleutherAI/hendrycks_math")
            )
        return self._subgroups

    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format MATH problem.

        The 'problem' field contains the question text.
        """
        problem = item["problem"]
        return f"{problem}"

    def extract_metadata(
        self, item: Dict[str, Any], subgroup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata including difficulty level and solution."""
        metadata = super().extract_metadata(item, subgroup)

        # MATH problems have difficulty levels
        if "level" in item:
            metadata["difficulty"] = item["level"]

        # Store solution separately (not in embedding)
        if "solution" in item:
            metadata["solution"] = item["solution"]

        metadata["question_type"] = "math_problem"

        # The 'type' field contains the category name
        if "type" in item:
            metadata["category_from_data"] = item["type"]

        return metadata

    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """Sample n questions from a specific MATH category."""
        # Load the category-specific dataset
        category_dataset = self.load_category_dataset(subgroup)

        total_available = len(category_dataset)
        if total_available < n:
            print(
                f"Warning: {subgroup} has only {total_available} samples, requested {n}"
            )
            indices = list(range(total_available))
        else:
            indices = random.sample(range(total_available), n)

        samples = []
        for idx in indices:
            item = category_dataset[idx]
            sample = self._create_sample(item, idx, subgroup)
            samples.append(sample)

        return samples

    def _sample_from_dataset(self, dataset: Dataset, n: int) -> List[BenchmarkSample]:
        """Not used for MATH (has subgroups/categories)."""
        raise NotImplementedError("MATH has category-based subgroups")

    def sample_questions(
        self,
        n_per_subgroup: int = 10,
        total_samples: int = 30,
    ) -> List[BenchmarkSample]:
        """
        Override to handle MATH's multiple config structure.
        """
        subgroups = self.get_subgroups()
        samples = []

        for category in subgroups:
            samples.extend(self._sample_from_subgroup(None, category, n_per_subgroup))

        return samples
