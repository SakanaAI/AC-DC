"""
MMLU benchmark loader.
"""

from typing import Any, Dict, List, Optional
from datasets import Dataset, load_dataset
import random
import logging

from benchmark_question_db.loaders.base import BenchmarkLoader, BenchmarkSample
from benchmark_question_db.loaders.invalid_filter import InvalidSampleFilter


class MMLULoader(BenchmarkLoader):
    """Loader for MMLU (Massive Multitask Language Understanding) dataset."""

    def __init__(self, random_seed: int = 42, include_mcq_options: bool = False):
        super().__init__("MMLU", random_seed, include_mcq_options)
        self._dataset = None
        self._subgroups = None
        self.invalid_filter = InvalidSampleFilter("MMLU")
        self.logger = logging.getLogger("MMLULoader")

    def load_dataset(self) -> Dataset:
        """Load MMLU dataset from HuggingFace."""
        if self._dataset is None:
            self._dataset = load_dataset("cais/mmlu", "all", split="test")
        return self._dataset

    def get_subgroups(self) -> Optional[List[str]]:
        """Get all subjects in MMLU."""
        if self._subgroups is None:
            dataset = self.load_dataset()
            self._subgroups = sorted(list(set(dataset["subject"])))
        return self._subgroups

    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format MMLU question.

        If include_mcq_options is True, format as:
            [question text]
            A) [choice 0]
            B) [choice 1]
            C) [choice 2]
            D) [choice 3]

        Otherwise, format as:
            [question text]
        """
        question = item["question"]

        # If not including MCQ options, return just the question
        if not self.include_mcq_options:
            return f"{question}"

        # Include MCQ options
        choices = item["choices"]
        formatted = f"{question}\n"
        choice_letters = ["A", "B", "C", "D"]

        for i, choice in enumerate(choices):
            formatted += f"{choice_letters[i]}) {choice}\n"

        return formatted.strip()

    def extract_metadata(
        self, item: Dict[str, Any], subgroup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata including answer."""
        metadata = super().extract_metadata(item, subgroup)

        # Store answer (as index and letter)
        answer_index = item["answer"]
        choice_letters = ["A", "B", "C", "D"]
        metadata["answer_index"] = answer_index
        metadata["answer_letter"] = choice_letters[answer_index]
        metadata["answer_text"] = item["choices"][answer_index]
        metadata["question_type"] = "multiple_choice"
        metadata["num_choices"] = len(item["choices"])

        return metadata

    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """Sample n questions from a specific subject."""
        # Filter dataset for this subject
        subject_data = dataset.filter(lambda x: x["subject"] == subgroup)

        # Filter out invalid samples if filter is available
        if self.invalid_filter.has_invalid_samples():
            valid_indices = []
            for idx in range(len(subject_data)):
                if self.invalid_filter.is_valid_sample(subject_data[idx], subgroup):
                    valid_indices.append(idx)

            invalid_count = len(subject_data) - len(valid_indices)
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
            total_available = len(subject_data)
            if total_available < n:
                self.logger.warning(
                    f"{subgroup} has only {total_available} samples, requested {n}"
                )
                indices = list(range(total_available))
            else:
                indices = random.sample(range(total_available), n)

        # Create samples
        samples = []
        for idx in indices:
            item = subject_data[idx]
            sample = self._create_sample(item, idx, subgroup)
            samples.append(sample)

        return samples

    def _sample_from_dataset(self, dataset: Dataset, n: int) -> List[BenchmarkSample]:
        """Not used for MMLU (has subgroups), but implemented for completeness."""
        total_available = len(dataset)
        if total_available < n:
            indices = list(range(total_available))
        else:
            indices = random.sample(range(total_available), n)

        samples = []
        for idx in indices:
            item = dataset[idx]
            sample = self._create_sample(item, idx, None)
            samples.append(sample)

        return samples
