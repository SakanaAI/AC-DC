"""
GPQA benchmark loader.
"""

from typing import Any, Dict, List, Optional
from datasets import Dataset, load_dataset
import random
import logging

from benchmark_question_db.loaders.base import BenchmarkLoader, BenchmarkSample
from benchmark_question_db.loaders.invalid_filter import InvalidSampleFilter


class GPQALoader(BenchmarkLoader):
    """Loader for GPQA (Graduate-Level Google-Proof Q&A Benchmark) dataset."""

    def __init__(self, random_seed: int = 42, include_mcq_options: bool = False):
        super().__init__("GPQA", random_seed, include_mcq_options)
        self._dataset = None
        self._subgroups = None
        self.invalid_filter = InvalidSampleFilter("GPQA")
        self.logger = logging.getLogger("GPQALoader")

    def load_dataset(self) -> Dataset:
        """Load GPQA dataset from HuggingFace."""
        if self._dataset is None:
            self._dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        return self._dataset

    def get_subgroups(self) -> Optional[List[str]]:
        """Get all high-level domains in GPQA (Biology, Chemistry, Physics)."""
        if self._subgroups is None:
            dataset = self.load_dataset()
            self._subgroups = sorted(list(set(dataset["High-level domain"])))
        return self._subgroups

    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format GPQA question.

        If include_mcq_options is True, format as:
            [question text]
            A) [choice 0]
            B) [choice 1]
            C) [choice 2]
            D) [choice 3]

        Otherwise, format as:
            [question text]
        """
        question = item["Question"]

        # If not including MCQ options, return just the question
        if not self.include_mcq_options:
            return f"{question}"

        # Include MCQ options
        # Collect all 4 answers
        correct_answer = item["Correct Answer"]
        incorrect_answers = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]

        # Combine and shuffle (using hash of question for consistent shuffling)
        all_choices = [correct_answer] + incorrect_answers
        # Use hash of the question text as seed for this specific question
        # This ensures the same question always gets the same shuffle
        question_seed = hash(question) % (2**32)
        local_random = random.Random(question_seed + self.random_seed)
        local_random.shuffle(all_choices)

        formatted = f"{question}\n"
        choice_letters = ["A", "B", "C", "D"]

        for i, choice in enumerate(all_choices):
            formatted += f"{choice_letters[i]}) {choice}\n"

        return formatted.strip()

    def extract_metadata(
        self, item: Dict[str, Any], subgroup: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata including answer."""
        metadata = super().extract_metadata(item, subgroup)

        # Collect all 4 answers
        correct_answer = item["Correct Answer"]
        incorrect_answers = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]

        # Combine and shuffle with same logic as format_question
        all_choices = [correct_answer] + incorrect_answers
        question_seed = hash(item["Question"]) % (2**32)
        local_random = random.Random(question_seed + self.random_seed)
        local_random.shuffle(all_choices)

        # Find the index of the correct answer after shuffling
        answer_index = all_choices.index(correct_answer)
        choice_letters = ["A", "B", "C", "D"]

        metadata["answer_index"] = answer_index
        metadata["answer_letter"] = choice_letters[answer_index]
        metadata["answer_text"] = correct_answer
        metadata["question_type"] = "multiple_choice"
        metadata["num_choices"] = 4
        metadata["subdomain"] = item.get("Subdomain", "")
        metadata["difficulty"] = item.get("Writer's Difficulty Estimate", "")

        return metadata

    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """Sample n questions from a specific domain (Biology, Chemistry, or Physics)."""
        # Filter dataset for this domain
        domain_data = dataset.filter(lambda x: x["High-level domain"] == subgroup)

        # Filter out invalid samples if filter is available
        if self.invalid_filter.has_invalid_samples():
            valid_indices = []
            for idx in range(len(domain_data)):
                if self.invalid_filter.is_valid_sample(domain_data[idx], subgroup):
                    valid_indices.append(idx)

            invalid_count = len(domain_data) - len(valid_indices)
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
            total_available = len(domain_data)
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
            item = domain_data[idx]
            sample = self._create_sample(item, idx, subgroup)
            samples.append(sample)

        return samples

    def _sample_from_dataset(self, dataset: Dataset, n: int) -> List[BenchmarkSample]:
        """Not used for GPQA (has subgroups), but implemented for completeness."""
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
