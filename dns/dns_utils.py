import json
import logging
import dataclasses  # Import dataclasses
from typing import List, Dict, Optional
from datatypes import (
    DNSSolution,
    TaskMetric,
    ACDCSolution,
    ACDCTaskEvalDetail,
)  # Import ACDCTaskEvalDetail
from tasks.acdc_task import ACDCTask
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


# Checkpointing
def save_dns_archive(archive: List[DNSSolution], save_path: str) -> None:
    """Save DNS archive to JSON format.

    Args:
        archive: List of DNSSolution objects
        save_path: Path to save the archive JSON
    """
    archive_data = []

    for solution in archive:
        solution_data = {
            "model_path": solution.model_path,
            "fitness": float(
                solution.fitness
            ),  # Ensure it's a native Python float
            "skill_vector": solution.skill_vector,
            "rank": solution.rank if solution.rank is not None else None,
            "validation_quality": (
                float(solution.validation_quality)
                if solution.validation_quality is not None
                else None
            ),
        }
        archive_data.append(solution_data)

    try:
        with open(save_path, "w") as f:
            json.dump(archive_data, f, indent=2)
        logger.info(
            f"Successfully saved archive with {len(archive)} solutions to {save_path}"
        )
    except Exception as e:
        logger.error(f"Error saving archive to {save_path}: {str(e)}")
        # Log the first solution's raw data for debugging
        if archive:
            logger.error(f"First solution raw data: {vars(archive[0])}")


def load_dns_archive(load_path: str) -> List[DNSSolution]:
    """Load DNS archive from JSON format."""
    with open(load_path, "r") as f:
        archive_data = json.load(f)

    archive = []
    for solution_data in archive_data:
        solution = DNSSolution(
            model_path=solution_data["model_path"],
            fitness=solution_data["fitness"],
            skill_vector=solution_data["skill_vector"],
            rank=solution_data.get("rank", None),
            validation_quality=solution_data.get("validation_quality", None),
        )
        archive.append(solution)

    return archive


def save_ac_dc_archive(
    archive: List[ACDCSolution], save_path: str, max_details_to_log: int = 5
) -> None:
    """Save AC/DC archive to JSON format, including truncated evaluation details.

    Args:
        archive: List of ACDCSolution objects.
        save_path: Path to save the archive JSON.
        max_details_to_log: Maximum number of AC/DC evaluation details to log per solution.
    """
    archive_data = []

    for solution in archive:
        # Convert solution to dict, handling potential None values and types
        solution_dict = dataclasses.asdict(solution)

        # Ensure fitness and validation_quality are native floats if not None
        if solution_dict.get("fitness") is not None:
            solution_dict["fitness"] = float(solution_dict["fitness"])
        if solution_dict.get("validation_quality") is not None:
            solution_dict["validation_quality"] = float(
                solution_dict["validation_quality"]
            )

        # Sort acdc_skill_vector by task number if it exists and is a dict
        acdc_skill_vector = solution_dict.get("acdc_skill_vector")
        if acdc_skill_vector is not None:

            def task_sort_key(item):
                key = item[0]
                # Extract the number after 'task_' (if present)
                import re

                match = re.search(r"task_(\d+)_", key)
                return int(match.group(1)) if match else float("inf")

            sorted_items = sorted(acdc_skill_vector.items(), key=task_sort_key)
            solution_dict["acdc_skill_vector"] = dict(sorted_items)

        # Truncate acdc_eval_details if it exists and exceeds the limit
        if solution_dict.get("acdc_eval_details") and max_details_to_log >= 0:
            details = solution_dict["acdc_eval_details"]
            if len(details) > max_details_to_log:
                logger.debug(
                    f"Truncating acdc_eval_details for {solution.model_path} from {len(details)} to {max_details_to_log}"
                )
                solution_dict["acdc_eval_details"] = details[
                    -max_details_to_log:
                ]
        elif (
            max_details_to_log < 0
        ):  # Allow logging all details if max_details_to_log is negative
            pass  # Keep all details
        else:
            # Ensure the key exists even if None or empty, for consistent loading
            solution_dict["acdc_eval_details"] = None

        archive_data.append(solution_dict)

    try:
        with open(save_path, "w") as f:
            json.dump(archive_data, f, indent=2)
        logger.info(
            f"Successfully saved archive with {len(archive)} solutions to {save_path}"
        )
    except Exception as e:
        logger.error(f"Error saving archive to {save_path}: {str(e)}")
        # Log the first solution's raw data for debugging
        if archive:
            logger.error(f"First solution raw data: {vars(archive[0])}")


def load_ac_dc_archive(load_path: str) -> List[ACDCSolution]:
    """Load AC/DC archive from JSON format."""
    with open(load_path, "r") as f:
        archive_data = json.load(f)

    archive = []
    for solution_data in archive_data:
        # Reconstruct ACDCTaskEvalDetail objects if present
        eval_details_data = solution_data.get("acdc_eval_details")
        reconstructed_details = None
        if isinstance(eval_details_data, list):
            reconstructed_details = []
            for detail_dict in eval_details_data:
                # Check if detail_dict is actually a dict before proceeding
                if isinstance(detail_dict, dict):
                    try:
                        reconstructed_details.append(
                            ACDCTaskEvalDetail(**detail_dict)
                        )
                    except TypeError as e:
                        logger.warning(
                            f"Skipping invalid ACDCTaskEvalDetail data: {detail_dict}. Error: {e}"
                        )
                else:
                    logger.warning(
                        f"Skipping non-dict item in acdc_eval_details: {detail_dict}"
                    )

        solution = ACDCSolution(
            model_path=solution_data["model_path"],
            fitness=solution_data["fitness"],
            acdc_skill_vector=solution_data.get("acdc_skill_vector", None),
            rank=solution_data.get("rank", None),
            validation_quality=solution_data.get("validation_quality", None),
            acdc_eval_details=reconstructed_details,  # Assign reconstructed details
        )
        archive.append(solution)

    return archive


# DNS Archive Utils
# DNS mode skill vector without AC/DC tasks
def create_skill_vector(
    task_metrics: Dict[str, TaskMetric], tasks: List[ACDCTask]
) -> List[bool]:
    """Create a binary skill vector from task metrics.

    Args:
        task_metrics: Dictionary mapping task names to their metrics
        tasks: List of task objects

    Returns:
        Binary skill vector representing task success/failure
    """
    skill_vector = []
    for task in tasks:
        if task.task_name in task_metrics and hasattr(
            task_metrics[task.task_name], "example_results"
        ):
            # Extract only the boolean 'correct' values from example results
            skill_vector.extend(
                [
                    (
                        example_result.correct
                        if hasattr(example_result, "correct")
                        else False
                    )
                    for _, example_result in sorted(
                        task_metrics[task.task_name].example_results.items()
                    )
                ]
            )
    return skill_vector


# QD mode skill vector without AC/DC tasks
def create_qd_skill_vector(task_metrics: Dict[str, TaskMetric]) -> List[bool]:
    """Create a binary skill vector from task metrics.

    Args:
        task_metrics: Dictionary mapping task names to their metrics

    Returns:
        Binary skill vector representing task success/failure
    """
    skill_vector = []
    for _, metric in sorted(task_metrics.items()):
        if hasattr(metric, "example_results"):
            # Extract example results in sorted order by example_id
            sorted_results = sorted(metric.example_results.items())
            for _, example_result in sorted_results:
                # Check if the example_result has a 'correct' attribute
                if hasattr(example_result, "correct"):
                    skill_vector.append(example_result.correct)
                else:
                    skill_vector.append(False)
    return skill_vector


def create_dns_solution(
    model_path: str,
    task_metrics: Dict[str, TaskMetric],
    tasks: List[ACDCTask],
    validation_quality: Optional[float] = None,
) -> DNSSolution:
    """Create a DNS solution from model and task metrics."""
    skill_vector = create_skill_vector(task_metrics, tasks)
    fitness = sum(skill_vector) / len(skill_vector)

    return DNSSolution(
        model_path=model_path,
        fitness=fitness,
        skill_vector=skill_vector,
        validation_quality=validation_quality,
    )


def create_ac_dc_solution(
    model_path: str,
    task_metrics: Optional[Dict[str, TaskMetric]],  # Metrics for standard tasks
    acdc_skill_vector: Optional[Dict[str, float]],  # Skill vector for AC/DC tasks
    avg_acdc_quality: Optional[float],  # Average quality for AC/DC tasks
    validation_quality: Optional[float] = None,
    acdc_eval_details: Optional[
        List[ACDCTaskEvalDetail]
    ] = None,  # Add eval details param
    is_gibberish: bool = False,
) -> ACDCSolution:
    """Create an AC/DC solution from model evaluation results."""
    # Calculate fitness based on available metrics
    total_quality = 0.0
    num_metrics = 0
    if task_metrics:
        qualities = [m.quality for m in task_metrics.values()]
        if qualities:
            total_quality += sum(qualities)
            num_metrics += len(qualities)
    if avg_acdc_quality is not None:
        total_quality += avg_acdc_quality
        num_metrics += 1  # Count avg_acdc_quality as one metric source

    fitness = (total_quality / num_metrics) if num_metrics > 0 else 0.0

    return ACDCSolution(
        model_path=model_path,
        fitness=fitness,
        acdc_skill_vector=acdc_skill_vector,  # Store the AC/DC skill vector
        validation_quality=validation_quality,
        acdc_eval_details=acdc_eval_details,  # Pass eval details
        is_gibberish=is_gibberish,
    )


def convert_acdc_to_dns_solution(
    acdc_solution: ACDCSolution,
    ordered_acdc_task_ids: List[str],
    threshold: float,
) -> DNSSolution:
    """
    Converts an ACDCSolution to a DNSSolution by creating a boolean skill vector.

    Args:
        acdc_solution: The ACDCSolution object to convert.
        ordered_acdc_task_ids: A list of AC/DC task IDs in a consistent order.
        threshold: The score threshold to consider an AC/DC task passed (True).

    Returns:
        A DNSSolution object with a boolean skill vector.
    """
    skill_vector = []
    acdc_skills = acdc_solution.acdc_skill_vector or {}  # Handle None case

    for task_id in ordered_acdc_task_ids:
        score = acdc_skills.get(
            task_id, 0.0
        )  # Default to 0.0 if task_id is missing
        skill_vector.append(score >= threshold)

    return DNSSolution(
        model_path=acdc_solution.model_path,
        fitness=acdc_solution.fitness,  # Use the existing fitness
        skill_vector=skill_vector,
        rank=acdc_solution.rank,  # Preserve rank if it exists
        validation_quality=acdc_solution.validation_quality,
    )


def compute_difficulty_weights(population: List[DNSSolution]) -> List[float]:
    """Compute difficulty weights for each task sample based on population performance.

    Args:
        population: List of DNS solutions

    Returns:
        List of difficulty weights
    """
    if not population or not population[0].skill_vector:
        return []

    num_tasks = len(population[0].skill_vector)
    failure_counts = [0] * num_tasks

    # Count failures for each task across the population
    for solution in population:
        for i, skill in enumerate(solution.skill_vector):
            if not skill:  # If the model failed this task
                failure_counts[i] += 1

    # Convert to difficulty weights (ratio of models that fail)
    population_size = len(population)
    difficulty_weights = [count / population_size for count in failure_counts]

    return difficulty_weights


def compute_hamming_distance(
    skill_vector1: List[bool], skill_vector2: List[bool]
) -> float:
    """Compute Hamming distance between two non-AC/DC skill vectors.

    Args:
        skill_vector1: First skill vector
        skill_vector2: Second skill vector

    Returns:
        Hamming distance between vectors
    """
    if len(skill_vector1) != len(skill_vector2):
        raise ValueError("Skill vectors must have the same length")
    return sum(1 for a, b in zip(skill_vector1, skill_vector2) if a != b)


def compute_dominated_novelty_score(
    solution: DNSSolution,
    fitter_solutions: List[DNSSolution],
    k_neighbors: int,
    dominated_score: float = 999.0,
    use_skill_ratio: bool = False,
    use_difficulty_weights: bool = False,
    difficulty_weights: Optional[List[float]] = None,
    skill_ratio_to_full: bool = False,
    len_subset_skill_vector: Optional[int] = None,
    fittest_model_found: bool = False,
) -> float:
    """Compute dominated novelty score for a solution.

    Args:
        solution: DNS solution
        fitter_solutions: List of fitter solutions
        k_neighbors: Number of neighbors to consider
        dominated_score: Score assigned to dominated solutions
        use_skill_ratio: Whether to use skill ratio for dominance
        use_difficulty_weights: Whether to use difficulty weights
        difficulty_weights: Optional list of difficulty weights
        skill_ratio_to_full: Whether skill ratio should be to full vector
        len_subset_skill_vector: Optional length of subset skill vector
        fittest_model_found: Whether the fittest model has been found
    Returns:
        Dominated novelty score
    """
    if not fitter_solutions:  # No fitter solutions found
        return dominated_score, True

    # Apply skill vector subsetting if enabled
    if len_subset_skill_vector is not None:
        solution_skill_vector = solution.skill_vector[:len_subset_skill_vector]
        fitter_skill_vectors = [
            sol.skill_vector[:len_subset_skill_vector]
            for sol in fitter_solutions
        ]
    else:
        solution_skill_vector = solution.skill_vector
        fitter_skill_vectors = [sol.skill_vector for sol in fitter_solutions]

    if use_skill_ratio:
        # Compute skill-based novelty scores against each fitter solution
        skill_scores = []
        for fitter_skill_vector in fitter_skill_vectors:
            if use_difficulty_weights and difficulty_weights is not None:
                # Adjust difficulty weights if using subset
                subset_difficulty_weights = difficulty_weights
                if len_subset_skill_vector is not None:
                    subset_difficulty_weights = difficulty_weights[
                        :len_subset_skill_vector
                    ]

                # Weight unique skills by difficulty
                unique_solved_weighted = sum(
                    subset_difficulty_weights[i]
                    for i, (weak, strong) in enumerate(
                        zip(solution_skill_vector, fitter_skill_vector)
                    )
                    if weak and not strong
                )

                if skill_ratio_to_full:
                    # Use total skill vector length as denominator
                    total_weighted = sum(subset_difficulty_weights)

                    if total_weighted == 0:
                        skill_scores.append(0.0)
                    else:
                        # Compute weighted ratio of unique skills to total skill vector length
                        skill_scores.append(
                            unique_solved_weighted / total_weighted * 100.0
                        )
                else:
                    # Weight unsolved by stronger by difficulty
                    total_unsolved_weighted = sum(
                        subset_difficulty_weights[i]
                        for i, strong in enumerate(fitter_skill_vector)
                        if not strong
                    )

                    if total_unsolved_weighted == 0:
                        # If stronger solution solves everything, give a skill score of zero
                        skill_scores.append(0.0)
                    else:
                        # Compute weighted ratio of unique skills to total unsolved
                        skill_scores.append(
                            unique_solved_weighted
                            / total_unsolved_weighted
                            * 100.0
                        )
            else:
                # Original unweighted calculation
                # Find examples that weaker solution solves but stronger doesn't
                unique_solved = sum(
                    1
                    for weak, strong in zip(
                        solution_skill_vector, fitter_skill_vector
                    )
                    if weak and not strong
                )

                if skill_ratio_to_full:
                    # Use total skill vector length as denominator
                    total_skills = len(solution_skill_vector)

                    # Compute ratio of unique skills to total skill vector length
                    skill_scores.append(unique_solved / total_skills * 100.0)
                else:
                    # Find total examples stronger solution doesn't solve
                    total_unsolved_by_stronger = sum(
                        1 for strong in fitter_skill_vector if not strong
                    )

                    if total_unsolved_by_stronger == 0:
                        # If stronger solution solves everything, give a skill score of zero as weaker solution outclassed
                        skill_scores.append(0.0)
                    else:
                        # Compute ratio of unique skills to total unsolved
                        skill_scores.append(
                            unique_solved / total_unsolved_by_stronger * 100.0
                        )

        # Sort scores and take k nearest
        skill_scores.sort()
        k = min(k_neighbors, len(skill_scores))
        if k == 0:
            return dominated_score, True

        # Return mean of k lowest skill scores
        return sum(skill_scores[:k]) / k, fittest_model_found
    else:
        # Original Hamming distance based computation
        distances = []
        for fitter_skill_vector in fitter_skill_vectors:
            dist = compute_hamming_distance(
                solution_skill_vector, fitter_skill_vector
            )
            distances.append(dist)

        # Sort distances and take k nearest
        distances.sort()
        k = min(k_neighbors, len(distances))
        if k == 0:
            return dominated_score, True

        # Return mean distance to k nearest neighbors
        return sum(distances[:k]) / k, fittest_model_found


def update_dns_archive_top_fitness(
    archive: List[DNSSolution],
    new_solutions: List[DNSSolution],
    dns_cfg: DictConfig,
) -> List[DNSSolution]:
    """Update DNS archive using top-fitness only selection.

    Args:
        archive: Current archive
        new_solutions: New solutions to consider adding
        dns_cfg: DNS configuration object

    Returns:
        Updated archive with top fitness solutions
    """
    logger = logging.getLogger(__name__)
    if not archive and not new_solutions:
        return []

    # Combine current archive and new solutions
    all_solutions = archive + new_solutions

    # Sort by fitness (descending - highest fitness first)
    all_solutions.sort(key=lambda x: x.fitness, reverse=True)

    # Keep top solutions up to archive size
    final_archive = all_solutions[: dns_cfg.population_size]

    logger.info(
        f"Top-fitness selection: kept {len(final_archive)} solutions with fitness range "
        f"{final_archive[-1].fitness:.4f} to {final_archive[0].fitness:.4f}"
    )

    return final_archive


def update_dns_archive(
    archive: List[DNSSolution],
    new_solutions: List[DNSSolution],
    dns_cfg: DictConfig,  # Changed cfg to dns_cfg
    len_subset_skill_vector: Optional[int] = None,
) -> List[DNSSolution]:
    """Update DNS archive with new solutions.

    Args:
        archive: Current archive
        new_solutions: New solutions to consider adding
        cfg: Hydra configuration object
        len_subset_skill_vector: Optional length of subset skill vector

    Returns:
        Updated archive
    """
    logger = logging.getLogger(__name__)
    if not archive and not new_solutions:
        return []

    # Check if we should use top-fitness selection
    if dns_cfg.get("use_top_fitness_selection", False):
        return update_dns_archive_top_fitness(archive, new_solutions, dns_cfg)

    # Combine current archive and new solutions
    all_solutions = archive + new_solutions

    # Compute difficulty weights if enabled
    difficulty_weights = None
    if dns_cfg.use_difficulty_weights:  # Use dns_cfg
        difficulty_weights = compute_difficulty_weights(all_solutions)
        logger.info(
            f"Computed difficulty weights: min={min(difficulty_weights):.4f}, "
            f"max={max(difficulty_weights):.4f}, "
            f"mean={sum(difficulty_weights)/len(difficulty_weights):.4f}"
        )

    # Compute fitness and novelty scores for all solutions
    solution_scores = []
    fittest_model_found = False
    for i, solution in enumerate(all_solutions):
        # Find all solutions with higher fitness
        if not fittest_model_found:
            # here, we make sure to always include the fittest model
            fitter_solutions = [
                s for s in all_solutions if s.fitness > solution.fitness
            ]
        else:
            # here, we consider cases where multiple solutions have the same fitness
            # but might have different skill vectors
            fitter_solutions = [
                s
                for s in all_solutions
                if s.fitness >= solution.fitness
                and s.model_path != solution.model_path
            ]

        # Compute dominated novelty score
        # TODO: Add comments for default parameters
        novelty_score, fittest_model_found = compute_dominated_novelty_score(
            solution,
            fitter_solutions,
            dns_cfg.k_neighbors,  # Use dns_cfg
            dns_cfg.dominated_score,  # Use dns_cfg
            dns_cfg.use_skill_ratio,  # Use dns_cfg
            dns_cfg.use_difficulty_weights,  # Use dns_cfg
            difficulty_weights,
            dns_cfg.skill_ratio_to_full,  # Use dns_cfg
            len_subset_skill_vector,
            fittest_model_found=fittest_model_found,
        )

        solution_scores.append((i, novelty_score))

    # Sort solutions by novelty score (descending)
    solution_scores.sort(key=lambda x: x[1], reverse=True)

    # Keep top solutions up to archive size
    # Use population_size from dns_cfg
    selected_indices = [
        idx for idx, _ in solution_scores[: dns_cfg.population_size]
    ]
    final_archive = [all_solutions[i] for i in selected_indices]

    # Removed skill dominance filtering loop as per simplified plan

    return final_archive
