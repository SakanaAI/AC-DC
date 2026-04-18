from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass, field  # Add field

from tasks.base import TaskMetric


@dataclass
class ACDCTaskEvalDetail:
    """Stores detailed results for a single ACDC task evaluation."""

    task_id: str
    instructions: str
    raw_output: str
    score: float  # Store the score as well for context


@dataclass
class ACDCMergeResult:
    save_path: str  # Path to the saved model
    task_metrics: Optional[Dict[str, TaskMetric]] = (
        None  # Metrics for standard tasks
    )
    acdc_skill_vector: Optional[Dict[str, float]] = (
        None  # Skill vector for ACDC tasks {task_id: score}
    )
    avg_acdc_quality: Optional[float] = (
        None  # Average quality across evaluated ACDC tasks
    )
    # Add field for detailed ACDC eval results (optional list of details)
    acdc_eval_details: Optional[List[ACDCTaskEvalDetail]] = field(
        default=None, repr=False
    )  # Don't include potentially long details in default repr

    # Add field for whether the model returns gibberish or not
    is_gibberish: bool = False


@dataclass
class DNSSolution:
    """Represents a solution in the DNS archive."""

    model_path: str  # Path to the model
    fitness: float  # Overall fitness (accuracy across all tasks)
    skill_vector: List[bool]
    rank: Optional[int] = None  # Domination rank (computed during sorting)
    validation_quality: Optional[float] = (
        None  # Validation quality (accuracy on validation set)
    )


@dataclass
class ACDCSolution:
    """Represents a solution in the AC/DC archive."""

    model_path: str  # Path to the model
    fitness: float  # Overall fitness (accuracy across all tasks)
    acdc_skill_vector: Optional[Dict[str, float]] = (
        None  # Skill vector for ACDC tasks {task_id: score}
    )
    rank: Optional[int] = None  # Domination rank (computed during sorting)
    validation_quality: Optional[float] = (
        None  # Validation quality (accuracy on validation set)
    )
    # Add field for detailed ACDC eval results (optional list of details)
    acdc_eval_details: Optional[List[ACDCTaskEvalDetail]] = field(
        default=None, repr=False
    )  # Don't include potentially long details in default repr

    # Add field for whether the model returns gibberish or not
    is_gibberish: bool = False


class ACDCArchiveData(TypedDict):
    dns_archive: List[ACDCSolution]
    dirs: Dict[str, str]