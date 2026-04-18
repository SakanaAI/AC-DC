"""
Base classes and interfaces for benchmark loaders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datasets import Dataset
import random


@dataclass
class BenchmarkSample:
    """
    Represents a single benchmark question/problem.

    Attributes:
        question_id: Unique identifier for the question
        content: Formatted question text (for embedding)
        metadata: Additional information (benchmark, subgroup, answer, etc.)
        raw_data: Original data from dataset (for reference)
    """

    question_id: str
    content: str  # The text to be embedded
    metadata: Dict[str, Any]
    raw_data: Dict[str, Any]


class BenchmarkLoader(ABC):
    """
    Abstract base class for benchmark dataset loaders.

    Each benchmark loader is responsible for:
    1. Loading the dataset from source
    2. Identifying subgroups/categories (if any)
    3. Sampling questions according to strategy
    4. Formatting questions for embedding
    """

    def __init__(
        self,
        benchmark_name: str,
        random_seed: int = 42,
        include_mcq_options: bool = False,
    ):
        """
        Initialize the benchmark loader.

        Args:
            benchmark_name: Name of the benchmark (e.g., "MMLU", "GSM8K")
            random_seed: Random seed for reproducible sampling
            include_mcq_options: Whether to include multiple-choice options in question formatting
        """
        self.benchmark_name = benchmark_name
        self.random_seed = random_seed
        self.include_mcq_options = include_mcq_options
        random.seed(random_seed)

    @abstractmethod
    def load_dataset(self) -> Dataset:
        """
        Load the dataset from source (e.g., HuggingFace).

        Returns:
            The loaded dataset
        """
        pass

    @abstractmethod
    def get_subgroups(self) -> Optional[List[str]]:
        """
        Get list of subgroups/categories in the dataset.

        Returns:
            List of subgroup names, or None if no subgroups
        """
        pass

    @abstractmethod
    def format_question(self, item: Dict[str, Any]) -> str:
        """
        Format a question for embedding.

        Args:
            item: Raw data item from the dataset

        Returns:
            Formatted question text (without answer)
        """
        pass

    def extract_metadata(self, item: Dict[str, Any], subgroup: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from a dataset item.

        Args:
            item: Raw data item from the dataset
            subgroup: Subgroup/category name if applicable

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "benchmark": self.benchmark_name,
        }
        if subgroup:
            metadata["subgroup"] = subgroup

        return metadata

    def sample_questions(
        self,
        n_per_subgroup: int = 10,
        total_samples: int = 30,
    ) -> List[BenchmarkSample]:
        """
        Sample questions from the dataset according to strategy.

        For datasets with subgroups: sample n_per_subgroup from each
        For datasets without subgroups: sample total_samples

        Args:
            n_per_subgroup: Number of samples per subgroup (if applicable)
            total_samples: Total samples if no subgroups

        Returns:
            List of BenchmarkSample objects
        """
        dataset = self.load_dataset()
        subgroups = self.get_subgroups()
        samples = []

        if subgroups:
            # Sample from each subgroup
            for subgroup in subgroups:
                samples.extend(
                    self._sample_from_subgroup(
                        dataset, subgroup, n_per_subgroup
                    )
                )
        else:
            # Sample total_samples from entire dataset
            samples.extend(self._sample_from_dataset(dataset, total_samples))

        return samples

    @abstractmethod
    def _sample_from_subgroup(
        self, dataset: Dataset, subgroup: str, n: int
    ) -> List[BenchmarkSample]:
        """
        Sample n questions from a specific subgroup.

        Args:
            dataset: The loaded dataset
            subgroup: Name of the subgroup to sample from
            n: Number of samples

        Returns:
            List of BenchmarkSample objects
        """
        pass

    @abstractmethod
    def _sample_from_dataset(
        self, dataset: Dataset, n: int
    ) -> List[BenchmarkSample]:
        """
        Sample n questions from the entire dataset (no subgroups).

        Args:
            dataset: The loaded dataset
            n: Number of samples

        Returns:
            List of BenchmarkSample objects
        """
        pass

    def _create_sample(
        self,
        item: Dict[str, Any],
        index: int,
        subgroup: Optional[str] = None,
    ) -> BenchmarkSample:
        """
        Create a BenchmarkSample from a dataset item.

        Args:
            item: Raw data item from the dataset
            index: Index in original dataset
            subgroup: Subgroup name if applicable

        Returns:
            BenchmarkSample object
        """
        # Generate question ID
        if subgroup:
            question_id = f"{self.benchmark_name.lower().replace(' ', '_')}_{subgroup}_{index}"
        else:
            question_id = f"{self.benchmark_name.lower().replace(' ', '_')}_{index}"

        # Format question for embedding
        content = self.format_question(item)

        # Extract metadata
        metadata = self.extract_metadata(item, subgroup)
        metadata["original_index"] = index

        return BenchmarkSample(
            question_id=question_id,
            content=content,
            metadata=metadata,
            raw_data=item,
        )
