from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from vllm import LLM
from dataclasses import dataclass


@dataclass 
class ExampleResult:
    """Result for a single example evaluation"""
    example_id: int
    correct: bool
    score: float
    details: Dict[str, Any]


@dataclass 
class ExampleResultScore:
    """Result for a single example evaluation"""
    example_id: int
    correct: bool
    score: float


@dataclass
class TaskMetric:
    quality: float
    bc_ids: Tuple[int, ...]
    example_results: Dict[int, ExampleResultScore]  # Maps example_id to its result


class BaseTask(object):

    def __init__(
        self,
        bc_num_dims: int,
        bc_min_vals: List[float],
        bc_max_vals: List[float],
        bc_grid_sizes: List[int],
    ) -> None:
        self.bc_num_dims = bc_num_dims
        self.bc_min_vals = bc_min_vals
        self.bc_max_vals = bc_max_vals
        self.bc_grid_sizes = bc_grid_sizes
        self._train_ids = []
        self._validation_ids = []
        self._task_ids = []
        # Default task name - subclasses should override this
        self.task_name = "base_task"

    def _load_vllm(self, llm: LLM) -> Any:
        """Load llm into appropriate model wrapper."""
        raise NotImplementedError()

    def get_q_and_bc(self, llm: LLM, data_split: str) -> TaskMetric:
        """Evaluate the LLM and return both quality and BC grid id."""
        raise NotImplementedError()
    
    def evaluate_example(self, llm: LLM, example_id: int) -> ExampleResult:
        """Evaluate a single example and return the result."""
        raise NotImplementedError()
    
    def evaluate_for_quality(self, llm: LLM, data_split: str, adaptive_sample_limit: Optional[int] = None) -> float:
        """Evaluate the LLM and return only the quality/accuracy value.
        
        This is a simplified version of get_q_and_bc that doesn't compute behavior characterization.
        
        Args:
            llm: The language model to evaluate
            data_split: Which data split to evaluate on ('train', 'validation', or 'all')
            adaptive_sample_limit: Optional limit on the number of examples to evaluate
            
        Returns:
            Float representing the quality/accuracy value
        """
        raise NotImplementedError()
    
    def get_example_ids(self, data_split: str) -> List[int]:
        """Get list of example IDs for the given data split."""
        if data_split == "train":
            return self._train_ids
        elif data_split == "validation":
            return self._validation_ids
        elif data_split == "all":
            return self._task_ids
        else:
            raise ValueError(f"Invalid data split: {data_split}")

    def _get_bin_id(self, bc_idx: int, metric: float) -> int:
        bins = np.linspace(
            self.bc_min_vals[bc_idx],
            self.bc_max_vals[bc_idx],
            self.bc_grid_sizes[bc_idx] + 1,
        )
        return min(
            max(0, np.digitize(metric, bins, right=True) - 1),
            self.bc_grid_sizes[0] - 1,
        )
