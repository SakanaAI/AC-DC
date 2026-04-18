"""
Utility for filtering invalid samples from benchmark datasets.
"""

import json
import logging
from typing import Any, Dict, List, Set
from pathlib import Path


class InvalidSampleFilter:
    """
    Utility to load and filter invalid samples from benchmark datasets.

    Invalid samples are loaded from JSON files in the invalid_samples_per_benchmark
    directory and used to filter out problematic questions before sampling.
    """

    def __init__(self, benchmark_name: str):
        """
        Initialize the invalid sample filter.

        Args:
            benchmark_name: Name of the benchmark (e.g., "MMLU", "MMLU_Pro", "BigBenchHard")
        """
        self.benchmark_name = benchmark_name
        self.invalid_samples_by_subgroup: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(f"InvalidSampleFilter.{benchmark_name}")
        self._load_invalid_samples()

    def _load_invalid_samples(self) -> None:
        """Load invalid samples from JSON file if it exists."""
        # Map benchmark names to file names
        file_mapping = {
            "MMLU": "mmlu.json",
            "MMLU_Pro": "mmlu_pro.json",
            "BigBenchHard": "bbh.json",
            "GPQA": "gpqa.json",
        }

        if self.benchmark_name not in file_mapping:
            self.logger.debug(f"No invalid samples file configured for {self.benchmark_name}")
            return

        # Construct path to invalid samples file
        file_path = Path(__file__).parent.parent / "invalid_samples_per_benchmark" / file_mapping[self.benchmark_name]

        if not file_path.exists():
            self.logger.debug(f"Invalid samples file not found: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                invalid_samples = json.load(f)

            # Group by subgroup and create set of invalid question texts
            for sample in invalid_samples:
                # Get subgroup name (field name varies by benchmark)
                subgroup = sample.get("subject")  # MMLU, MMLU_Pro, BBH use "subject"
                if subgroup is None:
                    subgroup = sample.get("category")  # Alternative field name

                if subgroup is None:
                    self.logger.warning(f"Sample missing subgroup/subject field: {sample.get('question', '')[:50]}")
                    continue

                if subgroup not in self.invalid_samples_by_subgroup:
                    self.invalid_samples_by_subgroup[subgroup] = set()

                # Just use the question text as the identifier
                # BBH invalid samples JSON uses "question" field
                question = sample.get("question", sample.get("input", ""))
                if question:
                    self.invalid_samples_by_subgroup[subgroup].add(question)

            total_invalid = sum(len(v) for v in self.invalid_samples_by_subgroup.values())
            self.logger.info(
                f"Loaded {total_invalid} invalid samples for {self.benchmark_name} "
                f"across {len(self.invalid_samples_by_subgroup)} subgroups"
            )

        except Exception as e:
            self.logger.error(f"Failed to load invalid samples from {file_path}: {e}")
            self.invalid_samples_by_subgroup = {}

    def is_valid_sample(self, item: Dict[str, Any], subgroup: str = None) -> bool:
        """
        Check if a sample is valid (not in invalid list).

        Args:
            item: Dataset item to check
            subgroup: Subgroup/category/subject name

        Returns:
            True if sample is valid, False if it should be filtered out
        """
        if not self.invalid_samples_by_subgroup:
            return True  # No invalid samples loaded

        if subgroup not in self.invalid_samples_by_subgroup:
            return True  # No invalid samples for this subgroup

        # Get the question text from the item
        # Try "question" first (MMLU, MMLU_Pro), fall back to "input" (BBH)
        question = item.get("question", item.get("input", ""))

        # For BBH: The input field contains both question and options like:
        # "Question text?\nOptions:\n(A) choice1\n(B) choice2..."
        # or "Question text? Options: (A) choice1 (B) choice2..."
        # We need to extract just the question part before "Options:"
        if "input" in item and "Options:" in question:
            # Split on either "\nOptions:" or " Options:"
            if "\nOptions:" in question:
                question = question.split("\nOptions:")[0]
            elif " Options:" in question:
                question = question.split(" Options:")[0]

        # Check if this question is in the invalid set
        return question not in self.invalid_samples_by_subgroup[subgroup]

    def get_invalid_count(self, subgroup: str = None) -> int:
        """
        Get count of invalid samples.

        Args:
            subgroup: Optional subgroup name. If None, returns total count.

        Returns:
            Number of invalid samples
        """
        if subgroup is None:
            return sum(len(v) for v in self.invalid_samples_by_subgroup.values())
        return len(self.invalid_samples_by_subgroup.get(subgroup, set()))

    def get_subgroups_with_invalid_samples(self) -> List[str]:
        """
        Get list of subgroups that have invalid samples.

        Returns:
            List of subgroup names
        """
        return list(self.invalid_samples_by_subgroup.keys())

    def has_invalid_samples(self) -> bool:
        """
        Check if any invalid samples were loaded.

        Returns:
            True if invalid samples exist
        """
        return len(self.invalid_samples_by_subgroup) > 0
