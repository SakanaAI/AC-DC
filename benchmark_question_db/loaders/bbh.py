"""
Big Bench Hard (BBH) benchmark loader.
"""

from typing import Any, Dict, List, Optional
from datasets import Dataset, load_dataset, get_dataset_config_names
import random
import logging

from benchmark_question_db.loaders.base import BenchmarkLoader, BenchmarkSample
from benchmark_question_db.loaders.invalid_filter import InvalidSampleFilter


class BBHLoader(BenchmarkLoader):
    """Loader for Big Bench Hard dataset."""

    def __init__(self, random_seed: int = 42, include_mcq_options: bool = False):
        super().__init__("BigBenchHard", random_seed, include_mcq_options)
        self._datasets = {}  # Cache for each task's dataset
        self._subgroups = None
        self.invalid_filter = InvalidSampleFilter("BigBenchHard")
        self.logger = logging.getLogger("BBHLoader")

    def load_dataset(self) -> Optional[Dataset]:
        """
        BBH has multiple configs (tasks), so we load them separately.
        This method returns None; use load_task_dataset instead.
        """
        return None

    def load_task_dataset(self, task: str) -> Dataset:
        """Load dataset for a specific BBH task."""
        if task not in self._datasets:
            self._datasets[task] = load_dataset("SaylorTwift/bbh", task, split="test")
        return self._datasets[task]

    def get_subgroups(self) -> Optional[List[str]]:
        """Get all tasks in BBH."""
        if self._subgroups is None:
            self._subgroups = sorted(get_dataset_config_names("SaylorTwift/bbh"))
        return self._subgroups

    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format BBH question.

        BBH questions are in the 'input' field (SaylorTwift/bbh dataset).
        The input may contain "Options:" with choices - we extract just the question part.
        """
        input_text = item["input"]

        # Remove "Options:" part if present
        # Handle both "\nOptions:" and " Options:" formats
        if "Options:" in input_text:
            if "\nOptions:" in input_text:
                input_text = input_text.split("\nOptions:")[0]
            elif " Options:" in input_text:
                input_text = input_text.split(" Options:")[0]

        return f"{input_text}"

    def extract_metadata(
        self, item: Dict[str, Any], subgroup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata including answer."""
        metadata = super().extract_metadata(item, subgroup)

        # BBH answers are in the 'target' field
        metadata["answer"] = item["target"]
        metadata["question_type"] = "reasoning"

        return metadata

    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """Sample n questions from a specific BBH task."""
        # Load the task-specific dataset
        task_dataset = self.load_task_dataset(subgroup)

        # Filter out invalid samples if filter is available
        if self.invalid_filter.has_invalid_samples():
            valid_indices = []
            for idx in range(len(task_dataset)):
                if self.invalid_filter.is_valid_sample(task_dataset[idx], subgroup):
                    valid_indices.append(idx)

            invalid_count = len(task_dataset) - len(valid_indices)
            if invalid_count > 0:
                self.logger.info(
                    f"Filtered out {invalid_count} invalid samples from {subgroup} "
                    f"({len(valid_indices)} valid samples remaining)"
                )

            total_available = len(valid_indices)
            if total_available < n:
                self.logger.warning(
                    f"{subgroup} has only {total_available} valid samples, requested {n}"
                )
                indices = valid_indices
            else:
                indices = random.sample(valid_indices, n)
        else:
            # No filtering - use original logic
            total_available = len(task_dataset)
            if total_available < n:
                self.logger.warning(
                    f"{subgroup} has only {total_available} samples, requested {n}"
                )
                indices = list(range(total_available))
            else:
                indices = random.sample(range(total_available), n)

        samples = []
        for idx in indices:
            item = task_dataset[idx]
            sample = self._create_sample(item, idx, subgroup)
            samples.append(sample)

        return samples

    def _sample_from_dataset(self, dataset: Dataset, n: int) -> List[BenchmarkSample]:
        """Not used for BBH (has subgroups/tasks)."""
        raise NotImplementedError("BBH has task-based subgroups")

    def sample_questions(
        self,
        n_per_subgroup: int = 10,
        total_samples: int = 30,
    ) -> List[BenchmarkSample]:
        """
        Override to handle BBH's multiple config structure.
        """
        subgroups = self.get_subgroups()
        samples = []

        for task in subgroups:
            samples.extend(self._sample_from_subgroup(None, task, n_per_subgroup))

        return samples
