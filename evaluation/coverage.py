import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import time
import glob
import re
import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.utils import (
    ### Processing of samples per benchmark ###
    process_mmlu_samples,
    process_ifeval_samples,
    process_gsm8k_samples,
    process_mmlu_pro_samples,
    process_arc_challenge_samples,
    process_bbh_cot_zeroshot_samples,
    process_hendrycks_math_samples,
    process_minerva_math_samples,
    process_gpqa_samples,
    process_humaneval_instruct_samples,
    process_mbpp_instruct_samples,
    process_aime_samples,
    ### Processing of model eval results ###
    process_model_eval_results,
    process_model_metric_details,
    ### Selection functions ###
    # Per gen selection for per gen best of N
    get_best_n_models_based_on_coverage,
    get_best_n_models_based_on_fitness,
    # Across entire archive selection for global best of N
    get_top_n_models_based_on_fitness_across_entire_archive,
    get_top_n_models_from_gen_with_highest_coverage,
    get_top_n_models_manual_gen_selection,
    get_top_n_models_based_on_global_skill_vector,
    get_top_n_models_randomly,
    ### Utils ###
    get_model_name_from_lm_harness_path,
)
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAIN_METRICS_PER_BENCHMARK = {
    "gsm8k_llama": "exact_match,flexible_extract",
    "ifeval": "prompt_level_loose_acc,none",
    "mmlu_cot_llama": "exact_match,strict_match",
}

SELECTION_METHODS_TO_FUNCTIONS = {
    "get_top_n_models_manual_gen_selection": get_top_n_models_manual_gen_selection,
    "get_top_n_models_based_on_fitness_across_entire_archive": get_top_n_models_based_on_fitness_across_entire_archive,
    "get_top_n_models_from_gen_with_highest_coverage": get_top_n_models_from_gen_with_highest_coverage,
    "get_top_n_models_based_on_global_skill_vector": get_top_n_models_based_on_global_skill_vector,
    "get_top_n_models_randomly": get_top_n_models_randomly,
}


def missing_model_exists(model_paths: list[str]) -> bool:
    """
    Check if there is at least one model path that is None.
    """
    for model_path in model_paths:
        if model_path is None:
            return True
    return False


def get_missing_models(model_paths: list[str], model_eval_paths: list[str]) -> set:
    """
    Get the models that are in model_paths but not in model_eval_paths.
    """
    model_names_eval = [
        get_model_name_from_lm_harness_path(model_eval_path)
        for model_eval_path in model_eval_paths
    ]
    model_names_paths = [model_path.split("/")[-1] for model_path in model_paths]
    return set(model_names_paths) - set(model_names_eval)


def get_model_eval_path(saved_model_path: str, base_eval_results_path: str):
    """
    Get the model eval results path for a given saved model path.
    """
    model_name = saved_model_path.split("/")[-1]

    model_results_path = os.path.join(base_eval_results_path, model_name)
    if os.path.exists(model_results_path):
        return model_results_path
    else:
        raise ValueError(f"Model results path {model_results_path} does not exist.")

    # fetch all the dirs in the base_eval_results_path
    # all_dirs = glob.glob(f"{base_eval_results_path}/*")
    # for dir in all_dirs:
    #     if model_name in dir:
    #         return dir
    # return None


def get_coverage_metrics(
    model_eval_paths: list[str],
    benchmark_name: str,
    main_metric: str,
    experiment_path: str,
    create_unique_model_names: bool = False,
):
    """
    Analyzes model evaluation results to calculate
    Individual Accuracy, Coverage Accuracy, Majority Vote Accuracy,
    Coverage Contributions, and Unique Contributions metrics.
    This function is used to analyze the coverage results for a set of models
    that are selected as the "top" models for one generation.

    Args:
        model_eval_paths: List of paths to the model evaluation results.
        benchmark_name: The name of the benchmark to analyze.
        main_metric: The main metric to use for the score.
        experiment_path: The directory to save the results to.
    Returns:
        dict: A dictionary containing the analysis results for the\
            coverage results for one generation in the format of:
        {
            "num_runs_analyzed": <int>,
            "num_unique_examples": <int>,
            "individual_accuracies": {
                "model_name": <float>,
                ...
            },
            "coverage_accuracy": <float>,
            "coverage_correct_count": <int>,
            "majority_vote_accuracy": <float>,
            "majority_vote_correct_count": <int>,
            "coverage_contributions": {
                "model_name": <int>,
                ...
            },
            "unique_contributions": {
                "model_name": <int>,
                "model_name": <int>,
                ...
            },
            "analyzed_models": <list[str]>,
            "coverage_contribution_percentages": {
                "model_name": <float>,
                ...
            },
            "unique_contribution_percentages": {
                "model_name": <float>,
                ...
            }
        }
    """

    # Validate that there are no duplicate model paths
    model_names = [
        get_model_name_from_lm_harness_path(path) for path in model_eval_paths
    ]
    unique_model_names = set(model_names)
    if len(unique_model_names) != len(model_names):
        duplicate_models = [name for name in model_names if model_names.count(name) > 1]
        raise ValueError(
            f"Duplicate models detected in model_eval_paths! "
            f"Expected {len(model_names)} unique models but got {len(unique_model_names)}. "
            f"Duplicate models: {set(duplicate_models)}. "
            f"This will cause incorrect coverage calculations with fewer models than expected. "
            f"All model paths: {model_eval_paths}"
        )

    ### Get sample_id to model scores for the benchmark
    sample_id_to_model_scores = {}
    for i, model_eval_path in enumerate(model_eval_paths):
        acc_key = main_metric.split(",")[0]
        filter = main_metric.split(",")[-1]
        ## First get the sample details for the benchmark for the model
        if "gsm8k" in benchmark_name:
            evaluation_details = process_gsm8k_samples(
                model_eval_path, gsm8k_version=benchmark_name, filter=filter
            )
        elif "ifeval" in benchmark_name:
            evaluation_details = process_ifeval_samples(model_eval_path, main_metric)
        elif "mmlu_cot_llama" in benchmark_name:
            is_llm_as_a_judge = "llm_as_a_judge" in benchmark_name
            evaluation_details = process_mmlu_samples(
                model_eval_path,
                acc_key=acc_key,
                is_llm_as_a_judge=is_llm_as_a_judge,
            )
        elif "mmlu_pro" in benchmark_name:
            is_llm_as_a_judge = "llm_as_a_judge" in benchmark_name
            evaluation_details = process_mmlu_pro_samples(
                model_eval_path,
                acc_key=acc_key,
                filter=filter,
                is_llm_as_a_judge=is_llm_as_a_judge,
            )
        elif "arc_challenge" in benchmark_name:
            evaluation_details = process_arc_challenge_samples(model_eval_path)
        elif "bbh_cot_zeroshot" in benchmark_name:
            is_llm_as_a_judge = "llm_as_a_judge" in benchmark_name
            evaluation_details = process_bbh_cot_zeroshot_samples(
                model_eval_path,
                acc_key=acc_key,
                filter=filter,
                is_llm_as_a_judge=is_llm_as_a_judge,
            )
        elif "hendrycks_math" in benchmark_name:
            evaluation_details = process_hendrycks_math_samples(model_eval_path)
        elif "minerva_math" in benchmark_name:
            evaluation_details = process_minerva_math_samples(model_eval_path)
        elif "gpqa" in benchmark_name:
            is_llm_as_a_judge = "llm_as_a_judge" in benchmark_name
            evaluation_details = process_gpqa_samples(
                model_eval_path,
                acc_key=acc_key,
                filter=filter,
                is_llm_as_a_judge=is_llm_as_a_judge,
            )
        elif "humaneval_instruct" in benchmark_name:
            evaluation_details = process_humaneval_instruct_samples(model_eval_path)
        elif "mbpp_instruct" in benchmark_name:
            evaluation_details = process_mbpp_instruct_samples(model_eval_path)
        elif "aime" in benchmark_name:
            evaluation_details = process_aime_samples(model_eval_path)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        ## Then, map the sample_id to the model scores
        model_name = get_model_name_from_lm_harness_path(model_eval_path)

        if create_unique_model_names:
            model_name = f"{model_name}_run_{i+1}"

        for sample_id, details in evaluation_details.items():
            if sample_id not in sample_id_to_model_scores:
                sample_id_to_model_scores[sample_id] = {}

            # Add the model scores to the sample_id_to_model_scores
            sample_id_to_model_scores[sample_id][model_name] = details["correct"]

        # Save evaluation details to a json file
        if experiment_path is not None:
            save_path = f"{experiment_path}/eval/model_eval_details/{benchmark_name}/{model_name}_eval_details.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(evaluation_details, f, indent=4)

    # Initialize counters for aggregate metrics
    num_models = len(model_eval_paths)
    num_samples = len(sample_id_to_model_scores)
    coverage_correct_count = 0
    majority_vote_correct_count = 0
    coverage_contributions = defaultdict(
        int
    )  # {model_name: count of correct answers contributed to Coverage}
    unique_contributions = defaultdict(
        int
    )  # {model_name: count of uniquely correct answers}

    # Analyze aggregated results per example
    for sample_id in sorted(
        list(sample_id_to_model_scores.keys())
    ):  # Sort for consistent processing order
        model_scores_for_sample = sample_id_to_model_scores.get(
            sample_id, {}
        )  # Dict: model_name -> correct_bool
        if not model_scores_for_sample:
            logger.warning(
                f"No model results found for sample ID {sample_id}, skipping."
            )
            continue

        # Check correctness for each run for this example
        # List of names of models that got this sample correct
        correct_model_names = []
        # List of True/False for all models for this sample
        all_model_correct_flags = []

        for model_name in sorted(list(model_scores_for_sample.keys())):
            is_correct = model_scores_for_sample.get(
                model_name, False
            )  # Assume False if model missing for this sample
            all_model_correct_flags.append(is_correct)
            if is_correct:
                correct_model_names.append(model_name)

        # --- Coverage Calculation ---
        if correct_model_names:  # If at least one model was correct
            coverage_correct_count += 1
            # Increment contribution count for each model that solved it
            for model_name in correct_model_names:
                coverage_contributions[model_name] += 1

            # --- Unique Contribution Calculation ---
            if len(correct_model_names) == 1:
                unique_solver_model_name = correct_model_names[0]
                unique_contributions[unique_solver_model_name] += 1

        # --- Majority Vote Calculation ---
        num_correct_models = sum(all_model_correct_flags)
        # Use num_models (total unique models found) as the denominator N
        if num_correct_models > num_models / 2:
            majority_vote_correct_count += 1

    # Calculate overall aggregate accuracies
    coverage_accuracy = (
        (coverage_correct_count / num_samples) if num_samples > 0 else 0.0
    )
    majority_vote_accuracy = (
        (majority_vote_correct_count / num_samples) if num_samples > 0 else 0.0
    )

    # --- Individual accuracy ---
    individual_accuracies = {}
    for model_path in model_eval_paths:
        model_name = get_model_name_from_lm_harness_path(model_path)
        individual_accuracies[model_name] = process_model_eval_results(
            model_path, {benchmark_name: main_metric}
        )[benchmark_name]

        # Get the model metric details for all benchmarks
        model_metric_details = process_model_metric_details(model_path, benchmark_name)
        # Save the model metric details to a json file
        if experiment_path is not None:
            save_path = f"{experiment_path}/eval/model_eval_details/{benchmark_name}/{model_name}_metric_details.json"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(model_metric_details, f, indent=4)

    # Filter out None values from individual accuracies before saving
    valid_individual_accuracies = {
        k: v for k, v in individual_accuracies.items() if v is not None
    }

    results_summary = {
        "num_models_analyzed": num_models,
        "num_unique_samples": num_samples,
        "individual_accuracies": dict(
            sorted(valid_individual_accuracies.items())
        ),  # Add individual accuracies read from files
        f"coverage_accuracy": coverage_accuracy,
        f"coverage_correct_count": coverage_correct_count,
        "majority_vote_accuracy": majority_vote_accuracy,
        "majority_vote_correct_count": majority_vote_correct_count,
        "coverage_contributions": dict(sorted(coverage_contributions.items())),
        "unique_contributions": dict(sorted(unique_contributions.items())),
        "analyzed_files": [f.split("/")[-1] for f in model_eval_paths],
        # "analyzed_files": sorted([f.split("/")[-1] for f in model_eval_paths]),
    }

    # Calculate Coverage contribution percentages relative to total Coverage correct count
    if coverage_correct_count > 0:
        contribution_percentages = {
            run: count / coverage_correct_count
            for run, count in coverage_contributions.items()
        }
        results_summary["coverage_contribution_percentages"] = dict(
            sorted(contribution_percentages.items())
        )

    # Calculate unique contribution percentages relative to total number of examples
    if num_samples > 0:
        unique_percentages = {
            run: round((count / num_samples) * 100, 2)
            for run, count in unique_contributions.items()
        }
        results_summary["unique_contribution_percentages"] = dict(
            sorted(unique_percentages.items())
        )

    # logger.info(
    #     f"Best-of-{num_models} Accuracy: {coverage_accuracy:.4f} ({coverage_correct_count}/{num_samples})"
    # )
    # logger.info(
    #     f"Majority Vote Accuracy: {majority_vote_accuracy:.4f} ({majority_vote_correct_count}/{num_samples})"
    # )
    # logger.info(
    #     f"Unique Contributions (%): {dict(sorted(unique_percentages.items()))}\n"
    # )

    return results_summary


def get_coverage_metrics_per_benchmark(
    model_eval_paths: list[str],
    main_metrics_per_benchmark: dict = {
        "gsm8k": "exact_match,flexible-extract",
        "ifeval": "prompt_level_loose_acc,none",
        "mmlu_generation": "acc,none",
    },
    experiment_path: str = None,
    create_unique_model_names: bool = False,
):
    """
    Analyzes model evaluation results to calculate
    Individual Accuracy, Coverage Accuracy, Majority Vote Accuracy,
    Coverage Contributions, and Unique Contributions metrics.
    This function is used to analyze the coverage results for a set of models
    that are selected as the "top" models for one generation.

    Args:
        model_eval_paths: List of paths to the model evaluation results.
        main_metrics_per_benchmark: A dictionary mapping benchmark names to the main metric to use for the score.
        experiment_dir: The directory to save the results to.
    Returns:
        dict: A dictionary containing the analysis results for the\
            coverage results for one generation in the format of:
        {
            <benchmark_name>: {
                results from `get_coverage_metrics`
            },
            <benchmark_name>: {
                ...
            },
            ...
        }
    """

    logger.info(
        f"Found {len(model_eval_paths)} models to analyze for benchmarks: {list(main_metrics_per_benchmark.keys())}"
    )

    # Log model names for debugging
    model_names = [
        get_model_name_from_lm_harness_path(path) for path in model_eval_paths
    ]
    logger.debug(f"Models being analyzed: {model_names}")

    # Check for duplicates - this will be caught by get_coverage_metrics, but double-check here
    if len(set(model_names)) != len(model_names):
        duplicate_models = [name for name in model_names if model_names.count(name) > 1]
        raise ValueError(
            f"Duplicate models found in selection! "
            f"{len(model_names)} total, {len(set(model_names))} unique. "
            f"Duplicates: {set(duplicate_models)}"
        )

    # Process model evaluation results for each benchmark
    results_per_benchmark = {}
    for benchmark, main_metric in main_metrics_per_benchmark.items():
        logger.info(f"Processing {benchmark} with main metric: {main_metric}")
        results_per_benchmark[benchmark] = get_coverage_metrics(
            model_eval_paths,
            benchmark,
            main_metric,
            experiment_path,
            create_unique_model_names,
        )

    return results_per_benchmark


def get_coverage_metrics_per_benchmark_for_relevant_gens(
    experiment_path: str,
    relevant_gens: list[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    n_models: int = 3,
    main_metrics_per_benchmark: dict = {
        "gsm8k": "exact_match,flexible-extract",
        "ifeval": "prompt_level_loose_acc,none",
        "mmlu_generation": "acc,none",
    },
    lm_harness_name: str = "lm_harness",
) -> tuple[dict, dict]:
    """
    Get the coverage metrics for all generations in the output directory.

    Args:
        experiment_path: Path to the experiment directory.
        relevant_gens: List of generations to analyze.
        n_models: Number of models to analyze for coverage.

    Returns:
        tuple[dict, dict]: Two dictionaries containing the analysis results for the\
            coverage results for all generations for selection based on\
            coverage and fitness in the format of:
        {
            <benchmark_name>: {
                <gen_num>: {
                    results from `get_coverage_metrics`
                },
                <gen_num>: ...
            },
            <benchmark_name>: ...
        }
    """
    logger.info(
        f"Experiment path: {experiment_path}"
        f"\nGetting coverage metrics for N={n_models} models."
        f"\nRelevant generations: {relevant_gens}"
        f"\nBenchmarks and main metrics: {main_metrics_per_benchmark}\n"
    )
    all_archive_paths = glob.glob(f"{experiment_path}/archives/*")
    base_eval_results_path = f"{experiment_path}/eval/{lm_harness_name}"

    ### Get the relevant model paths for each gen ######################
    model_paths_per_gen_coverage = {}
    model_paths_per_gen_fitness = {}
    missing_gens = set()
    missing_models = set()
    for archive_path in all_archive_paths:
        if archive_path.endswith("filtered.json"):
            continue

        gen_num_pattern = r"gen(\d+)"
        match = re.search(gen_num_pattern, archive_path)
        if not match:
            logger.warning(f"No gen number found in {archive_path}")
            continue
        archive_gen_num = int(match.group(1))
        if archive_gen_num in set(relevant_gens):
            ## get the model paths for the gen
            # based on coverage
            model_paths = get_best_n_models_based_on_coverage(archive_path, n_models)

            # get the model eval paths given the model paths
            model_eval_paths = [
                get_model_eval_path(model_path, base_eval_results_path)
                for model_path in model_paths
            ]

            # if there is even one model path that is None, skip the gen
            if missing_model_exists(model_eval_paths):
                logger.warning(
                    f"Skipping gen {archive_gen_num} because there is at "
                    "least one model without eval results."
                )
                missing_gens.add(archive_gen_num)
                missing_models.update(get_missing_models(model_paths, model_eval_paths))
                continue

            model_paths_per_gen_coverage[archive_gen_num] = model_eval_paths
            # based on fitness
            model_paths = get_best_n_models_based_on_fitness(archive_path, n_models)

            model_eval_paths = [
                get_model_eval_path(model_path, base_eval_results_path)
                for model_path in model_paths
            ]

            # if there is even one model path that is None, skip the gen
            if missing_model_exists(model_eval_paths):
                logger.warning(
                    f"Skipping gen {archive_gen_num} because there is at "
                    "least one model without eval results."
                )
                missing_gens.add(archive_gen_num)
                missing_models.update(get_missing_models(model_paths, model_eval_paths))
                continue

            model_paths_per_gen_fitness[archive_gen_num] = model_eval_paths

    # for convenience, sort the model_paths_per_gen_coverage and model_paths_per_gen_fitness
    model_paths_per_gen_coverage = dict(sorted(model_paths_per_gen_coverage.items()))
    model_paths_per_gen_fitness = dict(sorted(model_paths_per_gen_fitness.items()))

    logger.info(f"Skipped {len(missing_gens)} gens due to missing model eval results.")
    logger.warning(
        f"Skipped models: {missing_models} due to missing model eval results."
    )
    relevant_gens = [gen for gen in relevant_gens if gen not in missing_gens]

    ### Get the coverage metrics for each gen #########################
    results_per_benchmark_coverage = {}
    results_per_benchmark_fitness = {}
    logger.info(
        f"Getting coverage metrics for {len(relevant_gens)} generations "
        "for coverage and fitness."
    )
    for gen in relevant_gens:
        ### based on coverage
        model_paths = model_paths_per_gen_coverage[gen]
        results_per_benchmark = get_coverage_metrics_per_benchmark(
            model_eval_paths=model_paths,
            main_metrics_per_benchmark=main_metrics_per_benchmark,
            experiment_path=experiment_path,
        )

        # save the results
        for benchmark, results in results_per_benchmark.items():
            if benchmark not in results_per_benchmark_coverage:
                results_per_benchmark_coverage[benchmark] = {}
            results_per_benchmark_coverage[benchmark][gen] = results

        ### based on fitness
        model_paths = model_paths_per_gen_fitness[gen]
        results_per_benchmark = get_coverage_metrics_per_benchmark(
            model_eval_paths=model_paths,
            main_metrics_per_benchmark=main_metrics_per_benchmark,
            experiment_path=experiment_path,
        )
        # save the results
        for benchmark, results in results_per_benchmark.items():
            if benchmark not in results_per_benchmark_fitness:
                results_per_benchmark_fitness[benchmark] = {}
            results_per_benchmark_fitness[benchmark][gen] = results

    ### save the results in in <benchmark_name>/coverage/ and <benchmark_name>/fitness/
    for benchmark in results_per_benchmark_coverage.keys():
        # coverage/
        save_path = f"{experiment_path}/eval/coverage/{benchmark}/coverage/results_N{n_models}.json"
        logger.info(f"Saving results to {save_path}.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results_per_benchmark_coverage[benchmark], f, indent=4)

        # fitness/
        save_path = f"{experiment_path}/eval/coverage/{benchmark}/fitness/results_N{n_models}.json"
        logger.info(f"Saving results to {save_path}.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results_per_benchmark_fitness[benchmark], f, indent=4)

    logger.warning(f"Skipped gens: {missing_gens} due to missing model eval results.")
    logger.warning(
        f"Skipped models: {missing_models} due to missing model eval results."
    )

    return results_per_benchmark_coverage, results_per_benchmark_fitness


def get_coverage_metrics_per_benchmark_across_entire_archive(
    experiment_path: str,
    n_models: int = 3,
    main_metrics_per_benchmark: dict = {
        "gsm8k": "exact_match,flexible-extract",
        "ifeval": "prompt_level_loose_acc,none",
        "mmlu_generation": "acc,none",
    },
    selection_methods: list[dict[str, Any]] = None,
    lm_harness_name: str = "lm_harness",
):
    """
    Find the top N models using different selection methods across\
    the entire archive and then get the coverage metrics.

    Args:
        experiment_path: Path to the experiment directory.
        n_models: Number of models to select.
        main_metrics_per_benchmark: A dictionary mapping benchmark names to the main metric to use for the score.
        selection_methods: List of selection methods to use. Each selection method is a dictionary with the following keys:
            - "func_name": The name of the selection function.
            - "kwargs": A dictionary of keyword arguments to pass to the selection method.
            - "save_name": The name (directory name) of the selection method to use for saving the results.

    Returns:
        tuple[dict, dict]: Two dictionaries containing the analysis results for the\
            coverage results for selection based on\
            fitness across entire archive and fitness one per gen in the format of:
        {
            <benchmark_name>: {
                results from `get_coverage_metrics`
            },
            <benchmark_name>: {
                ...
            },
            ...
        }
    """
    assert selection_methods is not None, (
        "selection_methods must be provided. " "See the docstring for more details."
    )

    base_eval_results_path = f"{experiment_path}/eval/{lm_harness_name}"

    ### Get the coverage metrics for each selection method ############
    for selection_method in selection_methods:
        start_time = time.time()
        logger.info(
            f"Running coverage computation for {selection_method['func_name']} with N={n_models} models."
        )
        ### Get the selection function
        selection_function = SELECTION_METHODS_TO_FUNCTIONS.get(
            selection_method["func_name"]
        )
        if selection_function is None:
            raise ValueError(
                f"Selection function {selection_method['func_name']} not found. "
                f"Please select from the following: {list(SELECTION_METHODS_TO_FUNCTIONS.keys())}"
            )

        ### Get the paths to the top models
        top_model_paths = selection_function(
            experiment_path=experiment_path,
            n=n_models,
            **selection_method["kwargs"],
        )

        ### Get the model eval paths given the model paths
        model_eval_paths = [
            get_model_eval_path(model_path, base_eval_results_path)
            for model_path in top_model_paths
        ]
        # check if there is at least one model eval path that is None
        if any(model_eval_path is None for model_eval_path in model_eval_paths):
            logger.warning(
                f"Skipping {selection_method['func_name']} at N={n_models} because there is at least one model without eval results."
                f"\nSkipped models: {get_missing_models(top_model_paths, model_eval_paths)}"
            )
            continue

        ### Get the coverage metrics given the selection models
        results_per_benchmark = get_coverage_metrics_per_benchmark(
            model_eval_paths=model_eval_paths,
            main_metrics_per_benchmark=main_metrics_per_benchmark,
            experiment_path=experiment_path,
        )

        ### Save the results
        for benchmark, results in results_per_benchmark.items():
            save_path = f"{experiment_path}/eval/coverage/{benchmark}/{selection_method['save_name']}/results_N{n_models}.json"
            logger.info(f"Saving results to {save_path}.")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)

        end_time = time.time()
        logger.info(
            f"Time taken for {selection_method['func_name']} at N={n_models}: {(end_time - start_time):.2f} seconds.\n\n"
        )


def full_coverage_sweep_per_gen(
    experiment_path: str,
    max_n_models: int = 5,
    relevant_gens: list[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    main_metrics_per_benchmark: dict = {
        "gsm8k": "exact_match,flexible-extract",
        "ifeval": "prompt_level_loose_acc,none",
        "mmlu_generation": "acc,none",
    },
    lm_harness_name: str = "lm_harness",
):
    """
    Run a full coverage sweep for all generations in the experiment path.
    """
    logger.info(
        f"Running full coverage sweep for N={max_n_models} models."
        f"\nRelevant generations: {relevant_gens}"
        f"\nBenchmarks and main metrics: {main_metrics_per_benchmark}"
    )
    for n_models in range(1, max_n_models + 1):
        logger.info(f"Running coverage sweep for N={n_models} models.\n")
        get_coverage_metrics_per_benchmark_for_relevant_gens(
            experiment_path=experiment_path,
            n_models=n_models,
            relevant_gens=relevant_gens,
            main_metrics_per_benchmark=main_metrics_per_benchmark,
            lm_harness_name=lm_harness_name,
        )


def full_coverage_sweep_across_entire_archive(
    experiment_path: str,
    max_n_models: int = 5,
    relevant_gens: list[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    main_metrics_per_benchmark: dict = {
        "gsm8k": "exact_match,flexible-extract",
        "ifeval": "prompt_level_loose_acc,none",
        "mmlu_generation": "acc,none",
    },
    selection_methods: list[dict[str, Any]] = None,
    lm_harness_name: str = "lm_harness",
):
    """
    Run a full coverage sweep across the entire archive.

    Args:
        experiment_path: Path to the experiment directory.
        max_n_models: Maximum number of models to select.
        relevant_gens: List of generations to analyze.
        main_metrics_per_benchmark: A dictionary mapping benchmark names to the main metric to use for the score.
        selection_methods: List of selection methods to use. Each selection method is a dictionary with the following keys:
            - "func_name": The name of the selection function.
            - "kwargs": A dictionary of keyword arguments to pass to the selection method.
            - "save_name": The name (directory name) of the selection method to use for saving the results.
    """
    logger.info(
        f"Running full coverage sweep across the entire archive for N={max_n_models} models."
        f"\nRelevant generations: {relevant_gens}"
        f"\nBenchmarks and main metrics: {main_metrics_per_benchmark}"
    )

    for n_models in range(1, max_n_models + 1):
        logger.info(f"Running coverage sweep for N={n_models} models.\n")
        get_coverage_metrics_per_benchmark_across_entire_archive(
            experiment_path=experiment_path,
            n_models=n_models,
            main_metrics_per_benchmark=main_metrics_per_benchmark,
            selection_methods=selection_methods,
            lm_harness_name=lm_harness_name,
        )


def test_get_coverage_metrics_per_benchmark():
    # Example values
    model_eval_paths = [
        "outputs/2025-05-27/08-31-08/eval/lm_harness/outputs__2025-05-27__08-31-08__models__gen_0_ind_3",
        "outputs/2025-05-27/08-31-08/eval/lm_harness/outputs__2025-05-27__08-31-08__models__gen_0_ind_10",
    ]
    benchmarks = ["gsm8k", "ifeval"]
    main_metrics = [
        "exact_match,flexible-extract",
        "prompt_level_loose_acc,none",
    ]

    get_coverage_metrics_per_benchmark(
        model_eval_paths=model_eval_paths,
        main_metrics_per_benchmark={
            benchmark: main_metric
            for benchmark, main_metric in zip(benchmarks, main_metrics)
        },
    )


def test_get_coverage_metrics_per_benchmark_for_relevant_gens(
    experiment_path="outputs/2025-05-27/08-31-08",
    relevant_gens=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    n_models=3,
    main_metrics_per_benchmark=MAIN_METRICS_PER_BENCHMARK,
):
    """
    Test the get_coverage_metrics_per_benchmark_for_relevant_gens function.
    """

    get_coverage_metrics_per_benchmark_for_relevant_gens(
        experiment_path=experiment_path,
        relevant_gens=relevant_gens,
        n_models=n_models,
        main_metrics_per_benchmark=main_metrics_per_benchmark,
    )


def coverage_baselines(
    experiment_path,
    main_metrics_per_benchmark,
    model_names: dict[str, str] = {
        "big_model": "Meta-Llama-3-70B-Instruct",
        "control": "Meta-Llama-3-8B-Instruct",
        "expert_1": "Llama-3-8B-Instruct-Coding-Expert/Llama-3-8B-Instruct-Coding-Expert",
        "expert_2": "Meta-Llama-3-8B-Instruct/seed_43",
        "expert_3": "Meta-Llama-3-8B-Instruct_gsm8k_English/Meta-Llama-3-8B-Instruct_gsm8k_English",
    },
    lm_harness_name: str = "lm_harness",
):
    path_to_baselines = f"{experiment_path}/{lm_harness_name}"

    _expected_model_names = set(
        ["big_model", "control", "expert_1", "expert_2", "expert_3"]
    )
    assert (
        set(model_names.keys()) == _expected_model_names
    ), f"Model names dict keys {set(model_names.keys())} not found in {_expected_model_names}"

    ### Seed Models 3 Experts #################################################
    if (
        model_names["expert_1"] is not None
        and model_names["expert_2"] is not None
        and model_names["expert_3"] is not None
    ):
        model_paths = [
            f"{path_to_baselines}/{model_names['expert_1']}",
            f"{path_to_baselines}/{model_names['expert_2']}",
            f"{path_to_baselines}/{model_names['expert_3']}",
        ]

        results_per_benchmark = get_coverage_metrics_per_benchmark(
            model_paths, main_metrics_per_benchmark, experiment_path
        )

        for benchmark in results_per_benchmark.keys():
            save_path = f"{experiment_path}/coverage/{benchmark}/results_N3.json"
            logger.info(f"Saving results to {save_path}.")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results_per_benchmark[benchmark], f, indent=4)

    ### Single Big Model #####################################################
    if model_names["big_model"] is not None:
        model_paths = [
            f"{path_to_baselines}/{model_names['big_model']}",
        ]

        # Get the model size from the name (XXB) or (XXb)
        regex = r"(\d+)[B|b]"
        model_size = re.search(regex, model_names["big_model"]).group(1)
        logger.info(f"Model size: {model_size}B")

        results_per_benchmark = get_coverage_metrics_per_benchmark(
            model_paths, main_metrics_per_benchmark, experiment_path
        )

        for benchmark in results_per_benchmark.keys():
            save_path = f"{experiment_path}/coverage/{benchmark}/big_model_{model_size}B_N1.json"
            logger.info(f"Saving results to {save_path}.")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results_per_benchmark[benchmark], f, indent=4)

    ### Control Experiment BoN only 8B-Instruct ########################
    if model_names["control"] is not None:
        # model_paths = [
        #     f"{path_to_baselines}/{model_names['control']}/seed_43",
        #     f"{path_to_baselines}/{model_names['control']}/seed_44",
        #     f"{path_to_baselines}/{model_names['control']}/seed_45",
        #     f"{path_to_baselines}/{model_names['control']}/seed_46",
        #     f"{path_to_baselines}/{model_names['control']}/seed_47",
        #     f"{path_to_baselines}/{model_names['control']}/seed_48",
        #     f"{path_to_baselines}/{model_names['control']}/seed_49",
        #     f"{path_to_baselines}/{model_names['control']}/seed_50",
        # ]

        # find all seeds in the path_to_baselines
        seeds = [
            path.split("/")[-1]
            for path in glob.glob(f"{path_to_baselines}/{model_names['control']}/*")
        ]
        model_paths = [
            f"{path_to_baselines}/{model_names['control']}/{seed}" for seed in seeds
        ]

        for n in range(1, 9):
            results_per_benchmark = get_coverage_metrics_per_benchmark(
                model_eval_paths=model_paths[:n],
                main_metrics_per_benchmark=main_metrics_per_benchmark,
                experiment_path=experiment_path,
            )

            for benchmark in results_per_benchmark.keys():
                save_path = (
                    f"{experiment_path}/coverage/{benchmark}/control_results_N{n}.json"
                )
                logger.info(f"Saving results to {save_path}.")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(results_per_benchmark[benchmark], f, indent=4)


def get_all_unique_models_from_selection_methods(
    experiment_path: str,
    selection_methods: list[dict[str, Any]],
    max_n_models: int = 8,
    relevant_gens: list[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
) -> set[str]:
    """
    Get all unique model names from the selection methods config.

    Args:
        experiment_path: The path to the experiment directory.
        selection_methods: List of selection method configurations.
        max_n_models: Maximum number of models to select per method.
        relevant_gens: List of relevant generations.

    Returns:
        set[str]: Set of unique model names.
    """
    all_models = set()

    for method_config in selection_methods:
        func_name = method_config["func_name"]
        kwargs = method_config.get("kwargs", {})

        if func_name not in SELECTION_METHODS_TO_FUNCTIONS:
            logger.warning(f"Unknown selection function: {func_name}")
            continue

        try:
            # Get models for this selection method
            if func_name == "get_top_n_models_manual_gen_selection":
                # This function has a different signature
                models = SELECTION_METHODS_TO_FUNCTIONS[func_name](
                    experiment_path=experiment_path,
                    n=max_n_models,
                    relevant_gens=kwargs.get("relevant_gens", [relevant_gens[0]]),
                    selection_method=kwargs.get("selection_method", "coverage"),
                    coverage_optimization_method=kwargs.get(
                        "coverage_optimization_method", "greedy"
                    ),
                )
            else:
                # For other functions, use max_n_models as n
                models = SELECTION_METHODS_TO_FUNCTIONS[func_name](
                    experiment_path=experiment_path, n=max_n_models, **kwargs
                )

            # Print models for this selection method
            model_for_print = "\n".join(
                [model_path.split("/")[-1] for model_path in models]
            )
            logger.info(
                f"Models for {method_config.get('save_name', func_name)}:\n{model_for_print}\n"
            )

            # Extract model names from paths
            for model_path in models:
                if model_path:
                    model_name = model_path
                    all_models.add(model_name)

        except Exception as e:
            logger.warning(f"Error executing {func_name}: {e}")
            continue

    return all_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", "-e", type=str, required=True)
    parser.add_argument(
        "--baseline_eval",
        "-b",
        action="store_true",
        help="Whether to do baseline eval or archive eval.",
    )
    parser.add_argument(
        "--model_names_config",
        "-m",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--lm_harness_name",
        "-l",
        type=str,
        required=False,
        default="lm_harness",
        help="The name of the lm-harness experiment. Default is 'lm_harness'. This can be usefull, if the experiment has seperate lm_harness folders for different tasks.",
    )
    parser.add_argument(
        "--benchmark_metrics_config",
        "-bm",
        type=str,
        required=False,
        default="benchmarks_main.yaml",
    )
    parser.add_argument(
        "--selection_methods_config",
        type=str,
        required=False,
        default="selection_methods_main.yaml",
    )
    parser.add_argument("--max_n_models", type=int, required=False, default=8)
    parser.add_argument(
        "--relevant_gens",
        type=list[int],
        required=False,
        default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    )
    parser.add_argument(
        "--list_models_only",
        action="store_true",
        help="If set, only list all unique model names from selection methods config without running the full evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lm_harness_name = args.lm_harness_name
    experiment_path = args.experiment_path
    do_baseline_eval = args.baseline_eval
    max_n_models = args.max_n_models
    relevant_gens = args.relevant_gens
    model_names_config = args.model_names_config
    benchmark_metrics_config = args.benchmark_metrics_config
    selection_methods_config = args.selection_methods_config
    list_models_only = args.list_models_only

    base_path_to_config = "evaluation/pass@kModels_configs"
    with open(f"{base_path_to_config}/{benchmark_metrics_config}", "r") as f:
        main_metrics_per_benchmark = yaml.safe_load(f)

    if list_models_only:
        # Load selection methods config
        with open(f"{base_path_to_config}/{selection_methods_config}", "r") as f:
            selection_methods: list[dict[str, Any]] = yaml.safe_load(f)

        # Get all unique models
        unique_models = get_all_unique_models_from_selection_methods(
            experiment_path=experiment_path,
            selection_methods=selection_methods,
            max_n_models=max_n_models,
            relevant_gens=relevant_gens,
        )

        # Print unique model names and copy to clipboard
        print("Unique models from selection methods config:")
        output_lines = []
        for model_name in sorted(unique_models):
            output_lines.append(f"{model_name}")
            print(f"{model_name}")
        print(f"\nTotal unique models: {len(unique_models)}")

        return unique_models

    if do_baseline_eval:
        ### Best of N baselines ########################################
        assert (
            model_names_config is not None
        ), "Model names config is required for baseline eval."
        with open(
            f"{base_path_to_config}/baseline_model_names/{model_names_config}",
            "r",
        ) as f:
            model_names = yaml.safe_load(f)
        coverage_baselines(
            experiment_path=experiment_path,
            main_metrics_per_benchmark=main_metrics_per_benchmark,
            model_names=model_names,
            lm_harness_name=lm_harness_name,
        )
    else:
        with open(f"{base_path_to_config}/{selection_methods_config}", "r") as f:
            selection_methods: list[dict[str, Any]] = yaml.safe_load(f)

        full_coverage_sweep_across_entire_archive(
            experiment_path=experiment_path,
            max_n_models=max_n_models,
            relevant_gens=relevant_gens,
            main_metrics_per_benchmark=main_metrics_per_benchmark,
            selection_methods=selection_methods,
            lm_harness_name=lm_harness_name,
        )


if __name__ == "__main__":
    main()
