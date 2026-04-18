import os
import json
from itertools import combinations
from typing import List, Any
from pathlib import Path
import glob
import re
import random
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)


########################################################################
######################### Selection strategies #########################
########################################################################


def get_active_task_names_up_to_gen(
    experiment_dir: str,
    max_gen: int,
) -> set[str]:
    """Get all the tasks that were active up until that generation."""

    generated_tasks_dir = os.path.join(
        experiment_dir, "generated_tasks", "pool"
    )
    # Get all the active_pool_gen_X.json files in the experiment directory
    all_active_pool_files = glob.glob(
        os.path.join(generated_tasks_dir, "active_pool_gen_*.json")
    )

    # remove all the active_pool_files that have a gen number greater than max_gen
    all_active_pool_files = [
        file
        for file in all_active_pool_files
        if int(file.split("_")[-1].split(".")[0]) <= max_gen
    ]

    # Get the task names from the active_pool_files
    active_task_names = set()
    for file in all_active_pool_files:
        with open(file, "r") as f:
            all_task_dir_paths = json.load(f)

        for task_dir_path in all_task_dir_paths:
            task_name = task_dir_path.split("/")[-1]
            active_task_names.add(task_name)

    return active_task_names


def get_relevant_archive_models_and_skill_vectors(
    archive_path: str,
    models_dir: str = None,
    max_gen: int = None,
    do_max_gen_task_filtering: bool = False,
    include_seed_models: set[str] = "auto",
    exclude_seed_models: set[str] = None,
) -> List[dict]:
    """Get the relevant archive models and skill vectors."""
    # if archive_path is a directory, then assume they are global skill vectors
    # and get all the archive files in the directory
    if os.path.isdir(archive_path):
        assert (
            models_dir is not None
        ), "models_dir must be provided if archive_path is a directory"
        skill_vector_files = glob.glob(
            os.path.join(archive_path, "*_skill_vector.json")
        )
        skill_vector_files = sorted(skill_vector_files)
        assert len(skill_vector_files) > 0, (
            f"No skill vector files found in {archive_path}. "
            "Maybe you need to run `global_task_pool_eval.py` first."
        )

        assert (
            include_seed_models is None
            or isinstance(include_seed_models, (set, list))
            or include_seed_models == "auto"
        ), f"Invalid include_seed_models argument value: {include_seed_models}"

        if max_gen is not None and do_max_gen_task_filtering:
            # Get parent directory of archive path
            experiment_dir = os.path.dirname(archive_path)
            # Get all the tasks that were active up until that generation
            active_task_names = get_active_task_names_up_to_gen(
                experiment_dir=experiment_dir,
                max_gen=max_gen,
            )

        archive = []
        for skill_vector_file in skill_vector_files:
            if (
                max_gen is not None
                and int(skill_vector_file.split("/")[-1].split("_")[1])
                > max_gen
            ):
                continue
            with open(skill_vector_file, "r") as f:
                model_skill_vector = json.load(f)

            if max_gen is not None and do_max_gen_task_filtering:
                # Get the task names from the skill vector file
                task_names = list(model_skill_vector.keys())
                # Remove the tasks that were not active up until that generation
                model_skill_vector = {
                    task: model_skill_vector[task]
                    for task in task_names
                    if task in active_task_names
                }

            model_id = skill_vector_file.split("/")[-1].replace(
                "_skill_vector.json", ""
            )
            model_path = os.path.join(models_dir, model_id)
            fitness = sum(model_skill_vector.values()) / len(model_skill_vector)

            # Check if we want to strictly include a seed model
            # If include_seed_models is None, skip the seed models
            if include_seed_models is None and is_seed_model(model_path):
                continue

            # If include_seed_models is a set of model names, always include them
            elif isinstance(include_seed_models, (set, list)):
                if model_path.split("/")[-1] in include_seed_models:
                    fitness = (
                        np.inf
                    )  # Set the fitness to infinity to always include them

            elif include_seed_models == "auto":
                pass

            # Check if we want to strictly exclude a seed model
            # If exclude_seed_models is a set of model names, always exclude them
            if exclude_seed_models is not None and isinstance(
                exclude_seed_models, (set, list)
            ):
                if model_path.split("/")[-1] in exclude_seed_models:
                    continue

            archive.append(
                {
                    "model_path": model_path,
                    "acdc_skill_vector": model_skill_vector,
                    "fitness": fitness,
                }
            )
    else:
        with open(archive_path, "r") as f:
            archive = json.load(f)

    return archive


def get_best_n_models_based_on_coverage(
    archive_path,
    n,
    return_best_coverage=False,
    models_dir=None,
    coverage_optimization_method="greedy",
    max_gen: int = None,
    do_max_gen_task_filtering: bool = False,
    include_seed_models: set[str] = "auto",
    exclude_seed_models: set[str] = None,
):
    archive = get_relevant_archive_models_and_skill_vectors(
        archive_path=archive_path,
        models_dir=models_dir,
        max_gen=max_gen,
        do_max_gen_task_filtering=do_max_gen_task_filtering,
        include_seed_models=include_seed_models,
        exclude_seed_models=exclude_seed_models,
    )

    skill_vectors = [
        [model["acdc_skill_vector"][task] for task in model["acdc_skill_vector"]]
        for model in archive
    ]
    idx_to_model_path_map = {
        idx: model["model_path"] for idx, model in enumerate(archive)
    }

    assert len(skill_vectors) >= n, (
        f"The number of skill vectors ({len(skill_vectors)}) "
        f"must be greater than or equal to n ({n})."
    )

    # best_combination, best_coverage = optimal_model_selection(skill_vectors, n)
    if coverage_optimization_method == "greedy":
        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=skill_vectors,
            n=n,
            include_seed_models=include_seed_models,
            idx_to_model_path_map=idx_to_model_path_map,
        )
    elif coverage_optimization_method == "optimal":
        best_combination, best_coverage = optimal_model_selection(
            skill_vectors=skill_vectors,
            n=n,
            include_seed_models=include_seed_models,
            idx_to_model_path_map=idx_to_model_path_map,
        )
    else:
        raise ValueError(
            f"Invalid coverage optimization method: {coverage_optimization_method}"
        )

    assert len(best_combination) == n, (
        f"The number of models selected ({len(best_combination)}) "
        f"must be equal to n ({n})."
    )

    if return_best_coverage:
        return [
            idx_to_model_path_map[idx] for idx in best_combination
        ], best_coverage
    else:
        return [idx_to_model_path_map[idx] for idx in best_combination]


def get_best_n_models_based_on_fitness(
    archive_path: str,
    n: int,
    models_dir: str = None,
    max_gen: int = None,
    do_max_gen_task_filtering: bool = False,
    include_seed_models: set[str] = "auto",
    exclude_seed_models: set[str] = None,
) -> List[str]:
    """Select the best n models based on fitness from the archive."""

    archive = get_relevant_archive_models_and_skill_vectors(
        archive_path=archive_path,
        models_dir=models_dir,
        max_gen=max_gen,
        do_max_gen_task_filtering=do_max_gen_task_filtering,
        include_seed_models=include_seed_models,
        exclude_seed_models=exclude_seed_models,
    )

    # Sort by fitness and get top n
    sorted_models = sorted(archive, key=lambda x: x["fitness"], reverse=True)
    return [model["model_path"] for model in sorted_models[:n]]


def get_top_n_models_based_on_fitness_across_entire_archive(
    experiment_path: str,
    n: int,
    relevant_gens: list[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
) -> List[str]:
    """Select the best n models based on fitness from the entire archive."""
    model_id_to_model_path_and_fitness_and_skill_vector = (
        get_model_details_from_entire_relevant_archive(
            archive_path=f"{experiment_path}/archives",
            relevant_gens=relevant_gens,
        )
    )

    sorted_models = sorted(
        model_id_to_model_path_and_fitness_and_skill_vector.values(),
        key=lambda x: x["fitness"],
        reverse=True,
    )
    return [model["model_path"] for model in sorted_models[:n]]


def get_top_n_models_randomly(
    experiment_path: str,
    n: int,
    relevant_gens: list[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    seed: int = 42,
) -> List[str]:
    """Select the top n models randomly."""
    # Get all the model paths
    model_id_to_model_path_and_fitness_and_skill_vector = (
        get_model_details_from_entire_relevant_archive(
            archive_path=f"{experiment_path}/archives",
            relevant_gens=relevant_gens,
        )
    )
    model_details_list = list(
        model_id_to_model_path_and_fitness_and_skill_vector.values()
    )

    # IMPORTANT: Deduplicate models by model_path
    # A model can appear in multiple archive files (different generations)
    # We need to ensure each unique model is only selected once
    seen_model_paths = {}
    unique_model_details_list = []
    for model_details in model_details_list:
        model_path = model_details["model_path"]
        if model_path not in seen_model_paths:
            seen_model_paths[model_path] = True
            unique_model_details_list.append(model_details)
        else:
            logger.debug(f"Skipping duplicate model path: {model_path}")

    original_count = len(model_details_list)
    unique_count = len(unique_model_details_list)
    if original_count != unique_count:
        logger.warning(
            f"Found {original_count - unique_count} duplicate model paths in archive. "
            f"Using {unique_count} unique models for random selection."
        )

    # Set the random seed
    random.seed(seed)
    # Randomly shuffle the model paths
    random.shuffle(unique_model_details_list)

    # We need this, because some times, the model path in the archive
    # can be different from the model path in the models directory.
    def model_name_to_model_path(model_name: str, experiment_path: str) -> str:
        model_name = model_name.split("/")[-1]
        model_path = os.path.join(experiment_path, "models", model_name)
        return model_path

    # Return the top n models
    selected_models = [
        model_name_to_model_path(model["model_path"], experiment_path)
        for model in unique_model_details_list[:n]
    ]

    # Validate that we have exactly n unique models
    if len(set(selected_models)) != len(selected_models):
        duplicates = [m for m in selected_models if selected_models.count(m) > 1]
        raise AssertionError(
            f"BUG: Random selection produced duplicate models! "
            f"Expected {len(selected_models)} unique models but got {len(set(selected_models))}. "
            f"Duplicate models: {set(duplicates)}"
        )

    return selected_models


def get_top_n_models_based_on_global_skill_vector(
    experiment_path: str,
    n: int,
    max_gen: int = None,
    do_max_gen_task_filtering: bool = False,
    max_n: int = None,
    selection_method: str = "coverage",
    coverage_optimization_method: str = "greedy",
    include_seed_models: set[str] = "auto",
    exclude_seed_models: set[str] = None,
) -> List[str]:
    """Select the top n models based on the global skill vector.

    Args:
        experiment_path: The path to the experiment directory.
        n: The number of models to select.
        max_gen: The maximum generation upto which to select models from.
        max_n: The maximum number of models to select.
        selection_method: The method to use for selection. Can be "coverage" or "fitness".
        coverage_optimization_method: The method to use for coverage optimization. Can be "greedy" or "optimal" (brute force - very slow).
        include_seed_models: The set of seed models to include. If "auto", use the algorithm to select the seed models. If None, don't include any seed models. If a set of model names, always include them.
        exclude_seed_models: The set of seed models to exclude. If None, don't exclude any seed models. If a set of model names, exclude them.
    """
    if max_n and max_n < n:
        raise ValueError(
            f"max_n ({max_n}) must be greater than or equal to n ({n})"
        )

    # Get the global skill vector for each model
    skill_vector_dir = f"{experiment_path}/global_skill_vectors"
    assert os.path.isdir(
        skill_vector_dir
    ), f"Skill vector directory {skill_vector_dir} does not exist. Maybe you need to run `global_task_pool_eval.py` first."
    models_dir = f"{experiment_path}/models"

    if selection_method == "coverage":
        best_models = get_best_n_models_based_on_coverage(
            archive_path=skill_vector_dir,
            n=max_n if max_n else n,
            return_best_coverage=False,
            models_dir=models_dir,
            coverage_optimization_method=coverage_optimization_method,
            max_gen=max_gen,
            do_max_gen_task_filtering=do_max_gen_task_filtering,
            include_seed_models=include_seed_models,
            exclude_seed_models=exclude_seed_models,
        )
    elif selection_method == "fitness":
        best_models = get_best_n_models_based_on_fitness(
            archive_path=skill_vector_dir,
            n=max_n if max_n else n,
            models_dir=models_dir,
            max_gen=max_gen,
            do_max_gen_task_filtering=do_max_gen_task_filtering,
            include_seed_models=include_seed_models,
            exclude_seed_models=exclude_seed_models,
        )
    else:
        raise ValueError(f"Invalid selection method: {selection_method}")

    return best_models[:n]


def get_top_n_models_from_gen_with_highest_coverage(
    experiment_path: str,
    n: int,
    relevant_gens: list[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    max_n: int = None,
    coverage_optimization_method: str = "greedy",
) -> List[str]:
    """Select the top n models within the generation with the highest coverage."""
    if max_n and max_n < n:
        raise ValueError(
            f"max_n ({max_n}) must be greater than or equal to n ({n})"
        )

    models_and_coverage_per_gen = {}

    for gen_num in relevant_gens:
        best_models, best_coverage = get_best_n_models_based_on_coverage(
            archive_path=f"{experiment_path}/archives/gen{gen_num}_dns_archive.json",
            n=max_n if max_n else n,
            return_best_coverage=True,
            coverage_optimization_method=coverage_optimization_method,
        )
        models_and_coverage_per_gen[gen_num] = {
            "models": best_models,
            "coverage": best_coverage,
        }

    # Sort the gens by coverage
    sorted_gens = sorted(
        models_and_coverage_per_gen.keys(),
        key=lambda x: models_and_coverage_per_gen[x]["coverage"],
        reverse=True,
    )

    # Get the top n models within the best gen
    top_n_models = models_and_coverage_per_gen[sorted_gens[0]]["models"]
    # sort the top n models by name
    top_n_models = sorted(top_n_models)

    # Return the top n models
    return top_n_models[:n]


def get_top_n_models_manual_gen_selection(
    experiment_path: str,
    n: int,
    relevant_gens: list[tuple[int]],
    selection_method: str = "coverage",
    coverage_optimization_method: str = "greedy",
) -> List[str]:
    """Select the top n models within the a selected gens.
    
    Args:
        experiment_path: The path to the experiment directory.
        n: The number of models to select.
        relevant_gens: A list of tuples, where each tuple contains the gens from which to select the models.\
            Each entry should contain as many gens as the number of models to select.\
            E.g. [(5), (5, 10), (5, 10, 10), (5, 15, 20, 40)]
    
    Returns:
        list[str]: List of model paths.
    """
    assert len(relevant_gens) >= n, (
        f"The number of permutations in relevant_gens ({len(relevant_gens)}) "
        f"must be greater than or equal to n ({n})."
    )

    # Get the list of gens to use for N_models = n
    relevant_gens = relevant_gens[n - 1]  # E.g. (5, 10, 10)

    if isinstance(relevant_gens, int):
        relevant_gens = [relevant_gens]

    # Cunt the number of models to get per gen
    num_models_per_gen = Counter(relevant_gens)

    # Get the top models per gen
    best_models = []
    for gen_num in num_models_per_gen.keys():
        if selection_method == "coverage":
            best_models.extend(
                get_best_n_models_based_on_coverage(
                    archive_path=f"{experiment_path}/archives/gen{gen_num}_dns_archive.json",
                    n=num_models_per_gen[gen_num],
                    coverage_optimization_method=coverage_optimization_method,
                )
            )
        elif selection_method == "fitness":
            best_models.extend(
                get_best_n_models_based_on_fitness(
                    archive_path=f"{experiment_path}/archives/gen{gen_num}_dns_archive.json",
                    n=num_models_per_gen[gen_num],
                )
            )
        else:
            raise ValueError(f"Invalid selection method: {selection_method}")

    assert len(best_models) == n, (
        f"The number of models selected ({len(best_models)}) "
        f"must be equal to n ({n})."
    )

    return best_models


### Used for selection per gen ####
def get_top_models(
    path_to_archive: str, n: int, selection_method: str = "coverage"
) -> list[str]:
    """
    Get the top N models from the archive based on coverage or fitness.

    Returns:
        list[str]: List of absolute model paths.
    """
    if selection_method == "coverage":
        return get_best_n_models_based_on_coverage(
            path_to_archive, n, coverage_optimization_method="optimal"
        )
    elif selection_method == "fitness":
        return get_best_n_models_based_on_fitness(path_to_archive, n)
    else:
        raise ValueError(f"Invalid selection method: {selection_method}")


########################################################################
############ Processing of benchmark samples from lm-harness ###########
########################################################################


def process_gsm8k_samples(
    eval_output_dir: str,
    gsm8k_version: str = "gsm8k",
    filter: str = "flexible-extract",
) -> dict:
    """
    Process the GSM8K samples saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.
        filter: Select samples that used the filter type to extract the answer.
    Returns:
        dict[str, dict]: A dictionary of dictionaries containing the samples. In the format of:
        {
            0: {
                "correct": <true/ false>,
                "score": <1/ 0>,
                "problem": <problem>,
                "generation": <generation>,
                "prediction": <prediction>,
                "answer": <answer>
            }, ...
        }

    Note:
        The lm-harness samples are saved in the format of:
        {
            "doc_id": 0,
            "doc": {
                "question": <question>,
                "answer": <gt_answer>
            },
            "target": <gt_answer>,
            "arguments": {
                "gen_args_0": {
                    "arg_0": <full formatted prompt>,
                    "arg_1": {
                        "until": ["Question:", "</s>", "<|im_end|>"],
                        "do_sample": false,
                        "temperature": 0.0
                        }
                    }
                },
            "resps": [[<response_1>]],
            "filtered_resps": ["[invalid]"],
            "filter": "strict-match",
            "metrics": ["exact_match"],
            "doc_hash": <doc_hash>,
            "prompt_hash": <prompt_hash>,
            "target_hash": <target_hash>,
            "exact_match": <score>
        }, ...
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob(f"samples_{gsm8k_version}*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["filter"] == filter:
                    samples[sample["doc_id"]] = {
                        "correct": (
                            True if sample["exact_match"] == 1.0 else False
                        ),
                        "score": 1.0 if sample["exact_match"] == 1.0 else 0.0,
                        "problem": sample["doc"]["question"],
                        "generation": sample["resps"][0][0],
                        "prediction": sample["filtered_resps"][0],
                        "answer": sample["doc"]["answer"],
                    }
    return samples


def process_ifeval_samples(
    eval_output_dir: str, main_metric: str = "prompt_level_loose_acc,none"
) -> dict:
    """
    Process the IF-EVAL samples saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.
        main_metric: The main metric to use for the score.

    Returns:
        dict[str, dict]: A dictionary of dictionaries containing the samples. In the format of:
        {
            0: {
                "correct": <true/false>,
                "score": <1/0>,
                "problem": <prompt>,
                "generation": <generation>,
                "prediction": <prediction>,
                "answer": <target>
            }, ...
        }

    Note:
        The lm-harness samples are saved in the format of:
        {
            "doc_id": 0,
            "doc": {
                "key": <key>,
                "prompt": <prompt>,
                "instruction_id_list": <list[str]>,
                "kwargs": <list[dict]>
            },
            "target": <target>,
            "arguments": {
                "gen_args_0": {
                    "arg_0": <full formatted prompt>,
                    "arg_1": {
                        "until": [],
                        "do_sample": false,
                        "temperature": 0.0,
                        "max_gen_toks": <int>
                    }
                }
            },
            "resps": [[<response>]],
            "filtered_resps": [<filtered_response>],
            "filter": <filter_type>,
            "metrics": <list[str]>,
            "doc_hash": <hash>,
            "prompt_hash": <hash>,
            "target_hash": <hash>,
            "prompt_level_strict_acc": <bool>,
            "inst_level_strict_acc": <list[bool]>,
            "prompt_level_loose_acc": <bool>,
            "inst_level_loose_acc": <list[bool]>
        }
    """

    main_metric = main_metric.split(",")[0]

    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_ifeval_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                # inst_level_strict_acc
                inst_level_strict_acc = sum(
                    sample["inst_level_strict_acc"]
                ) / len(sample["inst_level_strict_acc"])
                # inst_level_loose_acc
                inst_level_loose_acc = sum(
                    sample["inst_level_loose_acc"]
                ) / len(sample["inst_level_loose_acc"])
                # prompt_level_strict_acc
                prompt_level_strict_acc = (
                    1.0 if sample["prompt_level_strict_acc"] else 0.0
                )
                # prompt_level_loose_acc
                prompt_level_loose_acc = (
                    1.0 if sample["prompt_level_loose_acc"] else 0.0
                )
                # main_metric_accuracy
                if main_metric == "inst_level_strict_acc":
                    main_metric_accuracy = inst_level_strict_acc
                elif main_metric == "inst_level_loose_acc":
                    main_metric_accuracy = inst_level_loose_acc
                elif main_metric == "prompt_level_strict_acc":
                    main_metric_accuracy = prompt_level_strict_acc
                elif main_metric == "prompt_level_loose_acc":
                    main_metric_accuracy = prompt_level_loose_acc
                else:
                    raise ValueError(f"Invalid main metric: {main_metric}")
                samples[sample["doc_id"]] = {
                    "correct": True if main_metric_accuracy else False,
                    "score": main_metric_accuracy,
                    "problem": sample["doc"]["prompt"],
                    "generation": sample["resps"][0][0],
                    "prediction": sample["filtered_resps"][0],
                    "answer": sample["target"],
                    "inst_level_strict_acc": inst_level_strict_acc,
                    "inst_level_loose_acc": inst_level_loose_acc,
                    "prompt_level_strict_acc": prompt_level_strict_acc,
                    "prompt_level_loose_acc": prompt_level_loose_acc,
                }
    return samples


def process_mmlu_samples(
    eval_output_dir: str,
    acc_key: str = "acc",
    is_llm_as_a_judge: bool = False,
) -> dict:
    """
    Process the MMLU samples saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.
        acc_key: The key to use for the accuracy.
        is_llm_as_a_judge: Whether the samples are from the LLM-as-a-judge experiment.
    Returns:
        dict[str, dict]: A dictionary of dictionaries containing the samples. In the format of:
        {
            0: {
                "correct": <true/false>,
                "score": <1/0>,
                "problem": <question>,
                "generation": <generation>,
                "prediction": <prediction>,
                "answer": <target>
            }, ...
        }

    Note:
        The lm-harness samples are saved in the format of:
        {
            "doc_id": 0,
            "doc": {
                "question": <question>,
                "subject": <subject>,
                "choices": <list[str]>,
                "answer": <int>
            },
            "target": <target>,
            "arguments": {
                "gen_args_0": {
                    "arg_0": <full formatted prompt>,
                    "arg_1": <choice>
                },
                ...
            },
            "resps": [[<response>]],
            "filtered_resps": [<filtered_response>],
            "filter": <filter_type>,
            "metrics": <list[str]>,
            "doc_hash": <hash>,
            "prompt_hash": <hash>,
            "target_hash": <hash>,
            "acc": <float>
        }
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_mmlu_cot_llama_*.jsonl"):
        if is_llm_as_a_judge and "llm_as_a_judge" not in file.name:
            continue
        elif not is_llm_as_a_judge and "llm_as_a_judge" in file.name:
            continue
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                subject = sample["doc"]["subject"]
                sample_id = f"{subject}_{sample['doc_id']}"
                samples[sample_id] = {
                    "correct": sample[acc_key] == 1.0,
                    "score": sample[acc_key],
                    "problem": sample["doc"]["question"],
                    "generation": sample["resps"][0][0],
                    "prediction": sample["filtered_resps"][0],
                    "answer": sample["target"],
                }
    return samples


def process_mmlu_pro_samples(
    eval_output_dir: str,
    acc_key: str = "exact_match",
    filter: str = "strict_match",
    is_llm_as_a_judge: bool = False,
) -> dict:
    """
    Process the MMLU-Pro samples saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.
        acc_key: The key to use for the accuracy.
        filter: The filter to use for the samples.
        is_llm_as_a_judge: Whether the samples are from the LLM-as-a-judge experiment.
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_mmlu_pro_llama_*.jsonl"):
        if is_llm_as_a_judge and "llm_as_a_judge" not in file.name:
            continue
        elif not is_llm_as_a_judge and "llm_as_a_judge" in file.name:
            continue
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                subject = sample["doc"]["category"]
                sample_id = f"{subject}_{sample['doc_id']}"
                samples[sample_id] = {
                    "correct": sample[acc_key] == 1.0,
                    "score": sample[acc_key],
                    "problem": sample["arguments"]["gen_args_0"]["arg_0"],
                    "generation": sample["resps"][0][0],
                    "prediction": sample["filtered_resps"][0],
                    "answer": sample["target"],
                }
    return samples


def process_arc_challenge_samples(eval_output_dir: str) -> dict:
    """
    Process the ARC-Challenge samples saved by lm-harness.
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_arc_challenge_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                samples[sample["doc_id"]] = {
                    "correct": sample["exact_match"] == 1.0,
                    "score": sample["exact_match"],
                    "problem": sample["arguments"]["gen_args_0"]["arg_0"],
                    "generation": sample["resps"][0][0],
                    "prediction": sample["filtered_resps"][0],
                    "answer": sample["target"],
                }
    return samples


def process_bbh_cot_zeroshot_samples(
    eval_output_dir: str,
    acc_key: str = "exact_match",
    filter: str = "flexible-extract",
    is_llm_as_a_judge: bool = False,
) -> dict:
    """
    Process the BBH-COT-ZeroShot samples saved by lm-harness.
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}

    def extract_group_name(file_path: str) -> str:
        """
        Extract the group name from the file path.
        the file name can be, e.g.,
        `samples_bbh_cot_zeroshot_word_sorting_2025-05-31T03-09-03.870778.jsonl`

        We want to get "word_sorting" from the file path.
        """
        # split by samples_bbh_cot_zeroshot_
        file_path = str(file_path)
        group_name = file_path.split("samples_bbh_cot_zeroshot_")[-1]
        # split by _ and remove the last part (the timestamp etc.)
        group_name = "_".join(group_name.split("_")[:-1])
        return group_name

    for file in eval_output_dir.glob("samples_bbh_cot_zeroshot_*.jsonl"):
        if is_llm_as_a_judge and "llm_as_a_judge" not in file.name:
            continue
        elif not is_llm_as_a_judge and "llm_as_a_judge" in file.name:
            continue
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                group_name = extract_group_name(file)
                sample_id = f"{group_name}_{sample['doc_id']}"
                if sample["filter"] == filter:
                    samples[sample_id] = {
                        "correct": sample[acc_key] == 1.0,
                        "score": sample[acc_key],
                        "problem": sample["arguments"]["gen_args_0"]["arg_0"],
                        "generation": sample["resps"][0][0],
                        "prediction": sample["filtered_resps"][0],
                        "answer": sample["target"],
                    }

    return samples


def process_hendrycks_math_samples(eval_output_dir: str) -> dict:
    """
    Process the Hendrycks Math samples saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.

    Returns:
        dict[str, dict]: A dictionary of dictionaries containing the samples. In the format of:
        {
            0: {
                "correct": <true/false>,
                "score": <1/0>,
                "problem": <problem>,
                "generation": <generation>,
                "prediction": <prediction>,
                "answer": <answer>
            }, ...
        }

    Note:
        The lm-harness samples are saved in the format of:
        {
            "doc_id": 0,
            "doc": {
                "problem": <problem>,
                "level": <level>,
                "type": <type>,
                "solution": <solution>,
                "answer": <answer>
            },
            "target": <target>,
            "arguments": {
                "gen_args_0": {
                    "arg_0": <full formatted prompt>,
                    "arg_1": {
                        "until": ["Problem:"],
                        "do_sample": false,
                        "temperature": 0.0
                    }
                }
            },
            "resps": [[<response>]],
            "filtered_resps": [<filtered_response>],
            "filter": <filter_type>,
            "metrics": ["exact_match"],
            "doc_hash": <hash>,
            "prompt_hash": <hash>,
            "target_hash": <hash>,
            "exact_match": <score>
        }
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_hendrycks_math_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                samples[sample["doc_id"]] = {
                    "correct": sample["exact_match"] == 1.0,
                    "score": sample["exact_match"],
                    "problem": sample["doc"]["problem"],
                    "generation": sample["resps"][0][0],
                    "prediction": sample["filtered_resps"][0],
                    "answer": sample["doc"]["answer"],
                }
    return samples


def process_minerva_math_samples(eval_output_dir: str) -> dict:
    """
    Process the Minerva Math samples saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.

    Returns:
        dict[str, dict]: A dictionary of dictionaries containing the samples. In the format of:
        {
            0: {
                "correct": <true/false>,
                "score": <1/0>,
                "problem": <problem>,
                "generation": <generation>,
                "prediction": <prediction>,
                "answer": <answer>
            }, ...
        }

    Note:
        The lm-harness samples are saved in the format of:
        {
            "doc_id": 0,
            "doc": {
                "problem": <problem>,
                "level": <level>,
                "type": <type>,
                "solution": <solution>,
                "answer": <answer>
            },
            "target": <target>,
            "arguments": {
                "gen_args_0": {
                    "arg_0": <full formatted prompt>,
                    "arg_1": {
                        "until": ["Problem:"],
                        "do_sample": false,
                        "temperature": 0.0
                    }
                }
            },
            "resps": [[<response>]],
            "filtered_resps": [<filtered_response>],
            "filter": <filter_type>,
            "metrics": ["math_verify"],
            "doc_hash": <hash>,
            "prompt_hash": <hash>,
            "target_hash": <hash>,
            "math_verify": <score>
        }
    """

    def extract_group_name(file_path: str) -> str:
        """
        Extract the group name from the file path.
        the file name can be, e.g.,
        `samples_bbh_cot_zeroshot_word_sorting_2025-05-31T03-09-03.870778.jsonl`

        We want to get "word_sorting" from the file path.
        """
        # split by samples_bbh_cot_zeroshot_
        file_path = str(file_path)
        group_name = file_path.split("samples_minerva_math_")[-1]
        # split by _ and remove the last part (the timestamp etc.)
        group_name = "_".join(group_name.split("_")[:-1])
        return group_name

    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_minerva_math_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                group_name = extract_group_name(file)
                sample_id = f"{group_name}_{sample['doc_id']}"
                samples[sample_id] = {
                    "correct": sample["math_verify"] == 1.0,
                    "score": sample["math_verify"],
                    "problem": sample["doc"]["problem"],
                    "generation": sample["resps"][0][0],
                    "prediction": sample["filtered_resps"][0],
                    "answer": sample["doc"]["answer"],
                }
    return samples


def process_gpqa_samples(
    eval_output_dir: str,
    acc_key: str = "exact_match",
    filter: str = "flexible-extract",
    is_llm_as_a_judge: bool = False,
) -> dict:
    """
    Process the GPQA samples saved by lm-harness.
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_gpqa_*.jsonl"):
        if is_llm_as_a_judge and "llm_as_a_judge" not in file.name:
            continue
        elif not is_llm_as_a_judge and "llm_as_a_judge" in file.name:
            continue
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["filter"] == filter:
                    samples[sample["doc_id"]] = {
                        "subdomain": sample["doc"]["Subdomain"],
                        "correct": sample[acc_key] == 1.0,
                        "score": sample[acc_key],
                        "problem": sample["arguments"]["gen_args_0"]["arg_0"],
                        "generation": sample["resps"][0][0],
                        "prediction": sample["filtered_resps"][0],
                        "answer": sample["target"],
                    }
    return samples


def process_humaneval_instruct_samples(
    eval_output_dir: str,
    acc_key: str = "pass@1",
    filter: str = "create_test",
) -> dict:
    """
    Process the HumanEval-Instruct samples saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.
        acc_key: The key to use for the accuracy.
        filter: The filter to use for the samples.
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_humaneval_instruct_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["filter"] == filter:
                    samples[sample["doc_id"]] = {
                        "correct": sample[acc_key] == 1.0,
                        "score": sample[acc_key],
                        "problem": sample["arguments"]["gen_args_0"]["arg_0"],
                        "generation": sample["resps"][0][0],
                        "prediction": sample["filtered_resps"][0][0],
                        "answer": sample["target"],
                    }
    return samples


def process_mbpp_instruct_samples(
    eval_output_dir: str,
    acc_key: str = "pass_at_1",
    filter: str = "extract_code",
) -> dict:
    """
    Process the MBPP-Instruct samples saved by lm-harness.
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_mbpp_instruct_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["filter"] == filter:
                    samples[sample["doc_id"]] = {
                        "correct": sample[acc_key] == 1.0,
                        "score": sample[acc_key],
                        "problem": sample["arguments"]["gen_args_0"]["arg_0"],
                        "generation": sample["resps"][0][0],
                        "prediction": sample["filtered_resps"][0][0],
                        "answer": sample["target"],
                    }
    return samples


def process_aime_samples(
    eval_output_dir: str,
    acc_key: str = "exact_match",
    filter: str = "flexible-extract",
) -> dict:
    """
    Process the AIME samples saved by lm-harness.
    """
    eval_output_dir = Path(eval_output_dir)
    samples = {}
    for file in eval_output_dir.glob("samples_aime_*.jsonl"):
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["filter"] == filter:
                    samples[sample["doc_id"]] = {
                        "correct": sample[acc_key] == 1.0,
                        "score": sample[acc_key],
                        "problem": sample["arguments"]["gen_args_0"]["arg_0"],
                        "generation": sample["resps"][0][0],
                        "prediction": sample["filtered_resps"][0],
                        "answer": sample["target"],
                    }
    return samples


########################################################################
########## General processing of model evaluation results ##############
########################################################################


def process_model_eval_results(
    eval_output_dir: str,
    main_metrics_per_benchmark: dict = {
        "gsm8k_llama": "exact_match,flexible_extract",
        "gsm8k": "exact_match,flexible-extract",
        "ifeval": "prompt_level_loose_acc,none",
        "mmlu_generation": "acc,none",
        "mmlu_pro_llama": "exact_match,strict_match",
        "arc_challenge_llama": "exact_match,strict_match",
        "bbh_cot_zeroshot": "exact_match,flexible-extract",
    },
) -> dict:
    """
    Process the model evaluation results saved by lm-harness.

    Args:
        eval_output_dir: The directory containing the evaluation output.
        main_metrics_per_benchmark: The main metrics to use for the score.

    Returns:
        dict[str, dict]: A dictionary of dictionaries containing the results. In the format of:
        {
            <benchmark_name>: {
                <main_metric>: <float>,
                ...
            }, ...
    """
    eval_output_dir = Path(eval_output_dir)
    results_per_benchmark = {}
    results_file = list(eval_output_dir.glob("results_*.json"))[0]
    with open(results_file, "r") as f:
        results = json.load(f)
    for benchmark, main_metric in main_metrics_per_benchmark.items():
        results_per_benchmark[benchmark] = results["results"][benchmark][
            main_metric
        ]
    return results_per_benchmark


def process_model_metric_details(
    eval_output_dir: str,
    benchmark_name: str = None,
) -> dict:
    """
     Process the model metric details saved by lm-harness.

    Args:
         eval_output_dir: The directory containing the evaluation output.
         benchmark_name: The name of the benchmark.
         model_name: The name of the model.

     Returns:
         dict[str, dict]: A dictionary of dictionaries containing the results. In the format of:
         {
             <benchmark_name>: {
                 <main_metric>: <float>,
                 ...
             }, ...
    """
    eval_output_dir = Path(eval_output_dir)
    model_metric_details = {}
    for file in eval_output_dir.glob("results_*.json"):
        with open(file, "r") as f:
            model_metric_details = json.load(f)

    # results
    if benchmark_name:
        results = model_metric_details["results"][benchmark_name]
    else:
        results = model_metric_details["results"]

    # groups
    if benchmark_name:
        if (
            "groups" in model_metric_details.keys()
            and benchmark_name in model_metric_details["groups"]
        ):
            if "mmlu_cot" in benchmark_name:
                groups = model_metric_details["groups"]
            elif "mmlu_pro" in benchmark_name:
                groups = {}
                for group_name in model_metric_details["results"]:
                    if benchmark_name in group_name:
                        groups[group_name] = model_metric_details["results"][
                            group_name
                        ]
            else:
                groups = model_metric_details["groups"][benchmark_name]
        else:
            groups = {benchmark_name: {}}
    else:
        groups = model_metric_details["groups"]

    # group_subtasks
    if benchmark_name:
        group_subtasks = model_metric_details["group_subtasks"][benchmark_name]
    else:
        group_subtasks = model_metric_details["group_subtasks"]

    return {
        "results": results,
        "groups": groups,
        "group_subtasks": group_subtasks,
    }


########################################################################
######################### Helper functions #############################
########################################################################


def optimal_model_selection(
    skill_vectors,
    n,
    include_seed_models="auto",
    idx_to_model_path_map=None,
):
    """
    [Brute Force] Select the best n models based on coverage.
    """

    if include_seed_models != "auto":
        raise NotImplementedError(
            "Optimal model selection does not support include_seed_models argument."
        )

    models = list(range(len(skill_vectors)))
    tasks = list(range(len(skill_vectors[0])))
    best_coverage = 0
    best_combination = None

    # Try all combinations of 'n' models
    for combo in combinations(models, n):
        # Calculate coverage
        coverage = 0
        for task in tasks:
            # Task is covered if any model in the combination succeeds
            if any(skill_vectors[model][task] == 1 for model in combo):
                coverage += 1

        # Update best if this combination has better coverage
        if coverage > best_coverage:
            best_coverage = coverage
            best_combination = combo

    return best_combination, best_coverage


def greedy_model_selection(
    skill_vectors,
    n,
    include_seed_models="auto",
    idx_to_model_path_map=None,
):
    """
    [Greedy] Select the best n models based on coverage using greedy approach.
    At each step, select the model that covers the most uncovered tasks.
    If no model adds new coverage, select the first available model.

    Args:
        skill_vectors: List of binary vectors representing model capabilities
        n: Number of models to select
        include_seed_models: The set of seed models to include. If "auto", use the algorithm to select the seed models. If None, don't include any seed models. If a set of model names, always include them.
        idx_to_model_path_map: A dictionary mapping the model indices to their paths.
    Returns:
        tuple: (selected_models, coverage_count)
    """
    num_models = len(skill_vectors)
    num_tasks = len(skill_vectors[0])

    # Convert to numpy for faster operations
    skills = np.array(skill_vectors)

    # Track which tasks are already covered
    covered_tasks = np.zeros(num_tasks, dtype=bool)

    seed_model_indices_to_skip = None
    selected_models = []

    if include_seed_models is None:
        # Get the seed model indices to make sure to not add them to the selection
        seed_model_indices_to_skip = set()
        for idx, model_path in idx_to_model_path_map.items():
            if is_seed_model(model_path):
                seed_model_indices_to_skip.add(idx)

    elif isinstance(include_seed_models, (set, list)):
        # Get the seed model indices to make sure to add them to the selection
        for idx, model_path in idx_to_model_path_map.items():
            model_name = model_path.split("/")[-1]
            if model_name in include_seed_models:
                # Add the seed model to the selection
                selected_models.append(idx)

                # Update the covered tasks
                model_tasks = skills[idx].astype(bool)
                covered_tasks |= model_tasks

            if len(selected_models) >= n:
                break

    elif include_seed_models == "auto":
        pass
    else:
        raise ValueError(
            f"Invalid include_seed_models argument value: {include_seed_models}"
        )

    while len(selected_models) < n:
        best_model = -1
        best_new_coverage = -1  # Start with -1 to handle zero coverage case

        # Try each remaining model
        for model_idx in range(num_models):
            if model_idx in selected_models:
                continue

            # (Maybe) Skip the seed models
            if (
                seed_model_indices_to_skip is not None
                and model_idx in seed_model_indices_to_skip
            ):
                continue

            # Calculate how many NEW tasks this model would cover
            model_tasks = skills[model_idx].astype(bool)
            new_coverage = np.sum(model_tasks & ~covered_tasks)

            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_model = model_idx

        # Add the best model found (guaranteed to find one if models remain)
        if best_model != -1:
            selected_models.append(best_model)
            # Update covered tasks
            covered_tasks |= skills[best_model].astype(bool)
        else:
            # This should never happen if n <= num_models, but safety check
            logger.warning(
                f"Could not find more models. Selected {len(selected_models)} out of {n}"
            )
            # Repeat the last model to fill the remaining slots
            selected_models.append(selected_models[-1])
            logger.warning(
                f"Repeating the last model to fill the remaining slots."
            )

    # Calculate final coverage
    total_coverage = np.sum(covered_tasks)

    return selected_models, total_coverage


def is_seed_model(model_path: str) -> bool:
    """
    Check if the model path is a seed model.
    Check if after "gen_<gen_num>_ind_" in the model path, there is a number.
    If there is no number, it is a seed model.

    E.g.
    "gen_0_ind_3" -> False
    "gen_0_ind_15" -> False
    "gen_0_ind_Qwen2.5-7B" -> True
    "gen_0_ind_Qwen2.5-7B-Instruct" -> True
    "gen_0_ind_Qwen2.5-7B-Instruct-Coding-Expert" -> True
    "gen_0_ind_Qwen2.5-7B-Instruct-Coding-Expert" -> True
    """
    model_name = model_path.split("/")[-1]
    match = re.match(r"gen_\d+_ind_(\d+)", model_name)
    if match:
        return False  # Not a seed model
    return True  # Is a seed model


def get_model_details_from_entire_relevant_archive(
    archive_path: str,
    relevant_gens: list[int],
) -> dict:
    """
    Get the model details from the entire relevant archive.

    Args:
        archive_path: The path to the archive.
        relevant_gens: The relevant gens to include.

    Returns:
        dict: A dictionary of model paths to their fitness and skill vector.
        {
            <model_path>: {
                "fitness": <fitness>,
                "acdc_skill_vector": <acdc_skill_vector>
            }, ...
        }
    """
    # Get all paths to the different archive files
    all_archive_paths = glob.glob(f"{archive_path}/*_archive.json")
    all_archive_paths = sorted(all_archive_paths)

    # Filter out the archive paths that are not in the relevant gens
    re_pattern_for_gen_num = r"gen(\d+)_dns_archive.json"
    relevant_archive_paths = []
    for path in all_archive_paths:
        match = re.search(re_pattern_for_gen_num, path)
        if match and int(match.group(1)) in relevant_gens:
            relevant_archive_paths.append(path)

    # Create a dictionary of all model paths to their respecitve fitness and skill vector
    model_paths_to_fitness_and_skill_vector = {}
    for archive_path in relevant_archive_paths:
        with open(archive_path, "r") as f:
            archive = json.load(f)
        match = re.search(re_pattern_for_gen_num, archive_path)
        if match:
            gen_num = int(match.group(1))
        else:
            raise ValueError(f"Could not find gen number in {archive_path}")
        for model in archive:
            model_paths_to_fitness_and_skill_vector[
                f"archive_{gen_num}_{model['model_path']}"
            ] = {
                "model_path": model["model_path"],
                "fitness": model["fitness"],
                "acdc_skill_vector": model["acdc_skill_vector"],
                "gen_num": gen_num,
            }

    return model_paths_to_fitness_and_skill_vector


def get_model_name_from_lm_harness_path(lm_harness_path: str) -> str:
    """
    Get the model name from the lm-harness path. E.g.
    outputs/2025-05-27/08-31-08/eval/lm_harness/outputs__2025-05-27__08-31-08__models__gen_0_ind_10
    or
    outputs/2025-05-27/08-31-08/eval/lm_harness/gen_0_ind_10
    """
    if not lm_harness_path:
        return None
    # split by __models__ and get the last part
    if "__models__" in lm_harness_path:
        model_name = lm_harness_path.split("__models__")[-1]
    else:
        model_name = lm_harness_path.split("/")[-1]
    return model_name


def remove_chat_template_from_question(
    question: str,
    start_header_str: str = "<|start_header_id|>",
    end_header_str: str = "<|end_header_id|>",
    bot_str: str = "<|begin_of_text|>",
    eot_str: str = "<|eot_id|>",
) -> str:
    """
    Remove the chat template from the question.
    """

    ### Remove few shot examples byt splitting based on the
    ## system and user tags
    # first, look for the system prompt
    pattern = f"{start_header_str}system{end_header_str}"
    question = question.split(pattern)[-1]

    # then, look for the user tag and remove it
    pattern = f"{start_header_str}user{end_header_str}"
    question = question.split(pattern)[-1]

    # then, look for the assistant tag and remove it
    pattern = f"{start_header_str}assistant{end_header_str}"
    question = question.replace(pattern, "")

    ### Remove the beginning of text and end of text
    question = question.replace(bot_str, "").replace(eot_str, "").strip()

    return question


def get_question_and_model_answer_from_sample_details(
    sample_details: dict,
    get_full_answer: bool = False,
) -> tuple[str, str]:
    """
    Get the question and answer from the sample details.
    """
    question = sample_details["arguments"]["gen_args_0"]["arg_0"]

    # Clean up chat template for question
    question = remove_chat_template_from_question(question).strip()

    if get_full_answer:
        answer = sample_details["resps"][0][0].strip()
    else:
        try:
            # For humaneval_instruct and mbpp_instruct, the filtered_resps is a nested list...
            # For whatever reason.
            if isinstance(sample_details["filtered_resps"][0], list):
                answer = sample_details["filtered_resps"][0][0].strip()
            else:
                answer = sample_details["filtered_resps"][0].strip()
        except Exception as e:
            logger.error(f"Error getting answer from sample details: {e}")
            logger.error(f"Sample details: {sample_details}")
            raise e

    return question, answer


def main():

    # Try out gsm8k processing
    # eval_output_dir = "outputs/2025-05-27/08-31-08/eval/lm_harness/outputs__2025-05-27__08-31-08__models__gen_1_ind_7"
    # samples = process_gsm8k_samples(eval_output_dir)
    # print(f"\nGSM8K samples: {len(samples)}")
    # print(samples[0])

    # # Try out ifeval processing
    # eval_output_dir = "outputs/2025-05-27/08-31-08/eval/lm_harness/outputs__2025-05-27__08-31-08__models__gen_1_ind_7"
    # samples = process_ifeval_samples(eval_output_dir)
    # print(f"\nIF-EVAL samples: {len(samples)}")
    # print(samples[0])

    # # Try out mmlu processing
    # eval_output_dir = "outputs/2025-05-27/08-31-08/eval/lm_harness/outputs__2025-05-27__08-31-08__models__gen_1_ind_7"
    # samples = process_mmlu_samples(eval_output_dir)
    # print(f"\nMMLU samples: {len(samples)}")
    # print(samples["machine_learning_0"])

    # Try out popolation selection
    # experiment_path = "outputs/2025-05-27/08-31-08"
    # n_models = 3
    # relevant_gens = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # top_n_models_based_on_fitness_across_entire_archive = (
    #     get_top_n_models_based_on_fitness_across_entire_archive(
    #         experiment_path=experiment_path,
    #         n=n_models,
    #         relevant_gens=relevant_gens,
    #     )
    # )
    # print(top_n_models_based_on_fitness_across_entire_archive)

    # process_model_metric_details
    eval_output_dir = "outputs/2025-05-27/08-31-08/eval/lm_harness/gen_0_ind_3"
    benchmark_name = "mmlu_pro_llama"
    model_metric_details = process_model_metric_details(
        eval_output_dir=eval_output_dir,
        benchmark_name=benchmark_name,
    )
    print(model_metric_details)


if __name__ == "__main__":
    main()
