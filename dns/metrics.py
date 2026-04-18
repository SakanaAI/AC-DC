from typing import Dict


def analyze_combined_coverage(
    combined_archive: Dict[str, Dict[str, bool]], top_k: int = 5
) -> Dict:
    """Analyze coverage metrics from a combined archive.

    Args:
        combined_archive: Combined archive mapping model paths to task success/failure
        top_k: Number of top models to analyze

    Returns:
        Dictionary containing coverage metrics
    """
    # Initialize coverage tracking
    coverage_stats = {
        "all_models": {
            "example_coverage": {},  # Maps example_id to whether any model passed it
            "total_examples": len(
                set().union(*[results.keys() for results in combined_archive.values()])
            ),
            "passed_examples": 0,
        },
        "top_models": {
            "example_coverage": {},  # Maps example_id to whether any top model passed it
            "total_examples": len(
                set().union(*[results.keys() for results in combined_archive.values()])
            ),
            "passed_examples": 0,
        },
    }

    # Initialize example coverage maps
    all_examples = set().union(
        *[results.keys() for results in combined_archive.values()]
    )
    for example_id in all_examples:
        coverage_stats["all_models"]["example_coverage"][example_id] = False
        coverage_stats["top_models"]["example_coverage"][example_id] = False

    # Get top k models by total number of passed examples
    model_scores = {
        model_path: sum(1 for passed in results.values() if passed)
        for model_path, results in combined_archive.items()
    }
    top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_model_paths = [model_path for model_path, _ in top_models]

    # Update coverage from all models
    for model_path, results in combined_archive.items():
        for example_id, passed in results.items():
            if passed:
                coverage_stats["all_models"]["example_coverage"][example_id] = True
                if model_path in top_model_paths:
                    coverage_stats["top_models"]["example_coverage"][example_id] = True

    # Calculate final statistics
    for category in ["all_models", "top_models"]:
        coverage_stats[category]["passed_examples"] = sum(
            1
            for passed in coverage_stats[category]["example_coverage"].values()
            if passed
        )
        # Assert that we have examples to avoid division by zero
        assert coverage_stats[category]["total_examples"] > 0, (
            f"Cannot compute coverage ratio for {category}: no examples found in combined_archive"
        )
        coverage_stats[category]["coverage_ratio"] = (
            coverage_stats[category]["passed_examples"]
            / coverage_stats[category]["total_examples"]
        )

    return coverage_stats


def compute_acdc_coverage_metrics(
    archive_data, tasks, cfg=None, threshold: float = 0.5, validation_tasks=None
) -> Dict:
    """Compute coverage metrics for wandb logging for AC/DC task vectors.

    Args:
        archive_data: Archive data containing dns_archive with AC/DC solutions
        tasks: List of task objects
        cfg: Optional configuration object (for backward compatibility)
        threshold: Score threshold to consider a task passed (default: 0.5)
        validation_tasks: Optional list of validation task names

    Returns:
        Coverage metrics dictionary for wandb logging
    """
    coverage_metrics = {}

    # Check if we have a valid archive
    if not archive_data or "dns_archive" not in archive_data or not archive_data["dns_archive"]:
        return coverage_metrics

    # Create combined coverage representation for AC/DC tasks
    combined_coverage = {}
    
    # Get all task_ids from all models to determine total task coverage
    all_task_ids = set()
    for solution in archive_data["dns_archive"]:
        if not solution or not hasattr(solution, "acdc_skill_vector") or not solution.acdc_skill_vector:
            continue
        all_task_ids.update(solution.acdc_skill_vector.keys())
        
        # Initialize combined coverage for this model
        combined_coverage[solution.model_path] = {}
        
    # If no task IDs found, return empty metrics
    if not all_task_ids:
        return coverage_metrics
        
    # Fill combined coverage with boolean pass/fail based on threshold
    for solution in archive_data["dns_archive"]:
        if not solution or not solution.acdc_skill_vector:
            continue
            
        for task_id in all_task_ids:
            # Default to 0.0 if task_id not in the skill vector
            score = solution.acdc_skill_vector.get(task_id, 0.0)
            combined_coverage[solution.model_path][task_id] = (score >= threshold)
    
    # Analyze the combined coverage with existing function
    top_k = 5
    combined_coverage_stats = analyze_combined_coverage(combined_coverage, top_k)
    
    if combined_coverage_stats:
        coverage_metrics.update({
            "acdc_coverage/combined/all_models/passed_ratio": combined_coverage_stats["all_models"]["coverage_ratio"],
            "acdc_coverage/combined/all_models/passed_examples": combined_coverage_stats["all_models"]["passed_examples"],
            "acdc_coverage/combined/all_models/total_examples": combined_coverage_stats["all_models"]["total_examples"],
            "acdc_coverage/combined/top_models/passed_ratio": combined_coverage_stats["top_models"]["coverage_ratio"],
            "acdc_coverage/combined/top_models/passed_examples": combined_coverage_stats["top_models"]["passed_examples"],
            "acdc_coverage/combined/top_models/total_examples": combined_coverage_stats["top_models"]["total_examples"],
        })
        
    # If validation tasks are present, include validation metrics for top models
    if validation_tasks or (cfg and hasattr(cfg, "validation_tasks") and cfg.validation_tasks):
        # Get the top 5 solutions for validation quality analysis
        valid_solutions = [s for s in archive_data["dns_archive"] if s and hasattr(s, "fitness")]
        sorted_solutions = sorted(valid_solutions, key=lambda x: x.fitness, reverse=True)[:5]
        
        # Track validation qualities for each top model
        for idx, solution in enumerate(sorted_solutions):
            if hasattr(solution, "validation_quality") and solution.validation_quality is not None:
                coverage_metrics[f"validation/top{idx+1}_model/quality"] = solution.validation_quality
        
        # If any models have validation_quality, find the best one
        validation_models = [s for s in sorted_solutions if hasattr(s, "validation_quality") and s.validation_quality is not None]
        if validation_models:
            best_validation_model = max(validation_models, key=lambda x: x.validation_quality)
            coverage_metrics["validation/best_validation_model/quality"] = best_validation_model.validation_quality
    
    return coverage_metrics
