import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import glob
import copy
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from create_main_best_of_N_plots import (
    get_best_of_N_results_per_benchmark_per_selection_method,
    get_baselines_results,
)

# from visualization.utils import (
from utils import (
    get_unique_contributions_relative_to_all_samples,
    multi_word_string_to_readable_string,
    compute_and_load_synth_task_to_tSNE_mapping,
    get_task_tSNE_graph_object,
    build_task_hover_text,
    mix_colors,
    get_single_answer_from_pop_data,
    shorten_model_name,
    COLOR_PALETTE as COLOR_MAP,
    create_interactive_generation_tree,
    extract_generation,
    create_equally_spaced_positions,
    compute_fitness_from_skill_vector,
    get_continuous_coevolution_scaling_law_data,
    get_continuous_coevolution_scaling_law_plot,
)

# COLOR_MAP = {
#     1: "#0068c9",  # Blue
#     2: "#83c9ff",  # Light Blue
#     3: "#7defa1",  # light green
#     4: "#ffabab",  # Peach
#     5: "#8c564b",  # brown
#     6: "#e377c2",  # pink
#     7: "#bcbd22",  # yellow-green
#     8: "#ff7f0e",  # orange
#     9: "#ff2b2b",  # Red
#     10: "#7f7f7f",  # gray
#     11: "#17becf",  # cyan
#     12: "#98df8a",  # light green

#     13: "#ff9999",  # light red
# }


@dataclass
class ModelExample:
    correct: bool
    score: float
    problem: str
    generation: str
    prediction: str
    answer: str


@dataclass
class BenchmarkData:
    model_examples: Dict[
        str, Dict[str, ModelExample]
    ]  # model_name -> task_id -> ModelExample
    task_ids: List[str]


class ExperimentData:
    def __init__(self, experiment_path: str):
        self.experiment_path = experiment_path
        self.existing_benchmarks = self._get_existing_benchmarks()
        # self.benchmark_data = self._get_benchmark_data()
        self.gen_to_selection_type_topN_to_model_names = (
            self._get_gen_to_selection_type_topN_to_model_names()
        )
        self.benchmark_to_metric_name = self._get_benchmark_to_metric_name()
        self.all_models_to_skill_vector = self._get_all_models_to_skill_vector()
        self.single_answer_from_pop_data = self._get_single_answer_from_pop_data()
        self.models_to_parent_models = self._get_model_to_parent_models()

    def get_max_top_N(self):
        """Get the maximum top N for the experiment."""
        # first_gen = list(self.gen_to_selection_type_topN_to_model_names.keys())[
        #     0
        # ]
        # all_top_Ns = set()
        # for selection_type in self.gen_to_selection_type_topN_to_model_names[
        #     first_gen
        # ]:
        #     all_top_Ns.update(
        #         self.gen_to_selection_type_topN_to_model_names[first_gen][
        #             selection_type
        #         ].keys()
        #     )
        best_of_N_dir = os.path.join(
            self.experiment_path,
            "eval",
            "best_of_N",
            self.existing_benchmarks[0],
        )
        all_selection_methods = glob.glob(os.path.join(best_of_N_dir, "*"))
        first_selection_method_dir = all_selection_methods[0]
        all_top_Ns = set(
            glob.glob(os.path.join(first_selection_method_dir, "results_N*.json"))
        )

        # filter the numbers from the file names
        reged_pattern = r"results_N(\d+).json"
        all_top_Ns = set(
            int(re.search(reged_pattern, file_name).group(1))
            for file_name in all_top_Ns
        )

        return max(all_top_Ns)

    def get_task_ids(self, benchmark: str):
        """Get the task ids for a given benchmark."""
        return self.benchmark_data[benchmark].task_ids

    def get_all_available_gens(self):
        """Get all available generations."""
        return list(self.gen_to_selection_type_topN_to_model_names.keys())

    def _get_benchmark_to_metric_name(self):
        """Get the benchmark to metric name."""
        return {
            "arc_challenge_llama": "exact_match,strict_match",
            "bbh_cot_zeroshot": "exact_match,flexible-extract",
            "gsm8k": "exact_match,flexible-extract",
            "gsm8k_llama": "exact_match,flexible_extract",
            "aime": "exact_match,flexible-extract",
            "minerva_math": "math_verify,none",
            "ifeval": "prompt_level_loose_acc,none",
            "humaneval_instruct": "pass@1,create_test",
            "mbpp_instruct": "pass_at_1,extract_code",
            "mmlu_cot_llama": "exact_match,strict_match",
            "mmlu_cot_llama_llm_as_a_judge": "acc,strict_match",
            "mmlu_pro_llama": "exact_match,strict_match",
            "mmlu_pro_llama_llm_as_a_judge": "acc,strict_match",
            "gpqa_main_cot_zeroshot": "exact_match,flexible-extract",
            "gpqa_diamond_cot_zeroshot": "exact_match,flexible-extract",
            "bbh_cot_zeroshot_llm_as_a_judge": "acc,strict_match",
            "gpqa_main_cot_zeroshot_llm_as_a_judge": "acc,strict_match",
        }

    def _get_single_answer_from_pop_data(self):
        """Get the single answer from pop data.

        Returns:
            dict[str, dict[str, dict[str, dict]]]: A dictionary of dictionaries containing the single answer from pop data.
            In the format of:
            {
                <task_force_selection_method>: {
                    <benchmark_name>: {
                        <model_group_name>: {
                            results: {
                                <selection_method>: <score>,
                                <selection_method>: <score>,
                                ...
                            },
                            model_distribution: {
                                <model_name>: {
                                    <selection_method>: <percentage>,
                                    <selection_method>: <percentage>,
                                    ...
                                },
                                <model_name>: {
                                    <selection_method>: <percentage>,
                                    <selection_method>: <percentage>,
                                    ...
                                ...
                            }
                        },
                        <model_group_name>: ...,
                    }, ...
                }, ...
        """
        single_answer_from_pop_dir = os.path.join(
            self.experiment_path, "eval", "single_answer_from_pop"
        )

        if not os.path.exists(single_answer_from_pop_dir):
            st.warning(
                f"Single answer from pop directory does not exist: {single_answer_from_pop_dir}"
                "Please run `evaluation/single_answer_from_pop_analysis.py` to generate the data."
            )
            return {}

        single_answer_from_pop_data = get_single_answer_from_pop_data(
            single_answer_from_pop_dir, self.existing_benchmarks
        )
        return single_answer_from_pop_data

    def _get_all_models_to_skill_vector(self):
        """Get the mapping from model name to (synthetic data) skill vector.

        Returns:
            dict[str, list[float]]: A dictionary of skill vectors.
            In the format of:
            {
                <model_name>: <skill_vector>,
                ...
            }
        """
        model_to_skill_vector = {}
        # find all skill vectors in experiment_path/global_skill_vectors
        skill_vectors_dir = os.path.join(self.experiment_path, "global_skill_vectors")
        for skill_vector_file in glob.glob(
            os.path.join(skill_vectors_dir, "*skill_vector.json")
        ):
            with open(skill_vector_file, "r") as f:
                skill_vector = json.load(f)
            model_name = os.path.basename(skill_vector_file).replace(
                "_skill_vector.json", ""
            )
            model_to_skill_vector[model_name] = skill_vector
        return model_to_skill_vector

    def _get_all_active_models(self):
        """Get all active model paths."""
        archive_files = glob.glob(
            os.path.join(self.experiment_path, "archives", "*.json")
        )
        all_active_models = set()
        for archive_file in archive_files:
            with open(archive_file, "r") as f:
                archive = json.load(f)
            for model in archive:
                all_active_models.add(model["model_path"].split("/")[-1])
        return all_active_models

    def _get_local_model_fitness_at_gen(self, model_name: str, gen: int):
        """Get the fitness of a model at a given generation."""
        archive_file = os.path.join(
            self.experiment_path,
            "archives",
            f"gen{gen}_dns_archive.json",
        )
        with open(archive_file, "r") as f:
            archive = json.load(f)

        for model in archive:
            if model["model_path"].split("/")[-1] == model_name:
                return model["fitness"]
        return None

    def _get_model_to_parent_models(self):
        """Get the mapping of each model to its parent models"""
        try:
            models_to_parent_models = {}
            # Get all model dirs
            all_active_models = self._get_all_active_models()
            # Get all model paths from experiment_path/models
            all_active_model_paths = glob.glob(
                os.path.join(self.experiment_path, "models", "*")
            )
            all_model_paths = glob.glob(
                os.path.join(self.experiment_path, "parent_models_mapping", "*")
            )
            for model_path in all_model_paths:
                # model_name = os.path.basename(model_path)
                model_name = os.path.basename(model_path).replace(".json", "")
                model_gen = extract_generation(model_name)
                if model_name not in all_active_models:
                    continue
                if model_name not in models_to_parent_models:
                    models_to_parent_models[model_name] = []

                # Get the parent model paths from model_path/parent_models.json
                parent_models_path = os.path.join(
                    self.experiment_path,
                    "parent_models_mapping",
                    f"{model_name}.json",
                )
                with open(parent_models_path, "r") as f:
                    parent_models = json.load(f)
                for parent_model in parent_models:
                    parent_model_name = parent_model.split("/")[-1]

                    if model_gen > 0:
                        parent_model_fitness = self._get_local_model_fitness_at_gen(
                            parent_model_name, model_gen - 1
                        )
                    else:
                        adjusted_seed_model_name = f"gen_0_ind_{parent_model_name}"
                        parent_model_fitness = self._get_local_model_fitness_at_gen(
                            adjusted_seed_model_name, 0
                        )

                    models_to_parent_models[model_name].append(
                        (parent_model_name, parent_model_fitness)
                    )
            return models_to_parent_models
        except Exception as e:
            st.error(f"Error getting model to parent models: {e}")
            return None

    def _get_gen_to_selection_type_topN_to_model_names(self):
        """Get the generation to selection type to top N model names.

        Returns:
            dict[str, dict[str, list[str]]]: A dictionary of dictionaries containing the top N model names.
            In the format of:
            {
                <generation>: {
                    <selection_type>: {
                        1: [<model_name>, ...],
                        2: [<model_name>, ...],
                        ...
                    }
                }, ...
        """

        gen_to_selection_type_topN_to_model_names = {}

        # Get the best of N directory
        best_of_N_dir = os.path.join(
            self.experiment_path,
            "eval",
            "best_of_N",
            self.existing_benchmarks[0],
        )

        ### Get the top model names for coverage and fitness
        for selection_type in ["coverage", "fitness"]:
            results_Nx_files = sorted(
                glob.glob(
                    os.path.join(best_of_N_dir, selection_type, "results_N*.json")
                )
            )
            for results_Nx_file in results_Nx_files:
                with open(results_Nx_file, "r") as f:
                    results_Nx = json.load(f)
                for gen_num, gen_results in results_Nx.items():
                    if int(gen_num) not in gen_to_selection_type_topN_to_model_names:
                        gen_to_selection_type_topN_to_model_names[int(gen_num)] = {}
                    if (
                        selection_type
                        not in gen_to_selection_type_topN_to_model_names[int(gen_num)]
                    ):
                        gen_to_selection_type_topN_to_model_names[int(gen_num)][
                            selection_type
                        ] = {}
                    n = gen_results["num_models_analyzed"]
                    gen_to_selection_type_topN_to_model_names[int(gen_num)][
                        selection_type
                    ][n] = list(gen_results["individual_accuracies"].keys())

        return gen_to_selection_type_topN_to_model_names

    def _get_existing_benchmarks(self):
        """Get the existing benchmarks."""
        return os.listdir(
            os.path.join(self.experiment_path, "eval", "model_eval_details")
        )

    def _load_single_model_eval_details(
        self, model_eval_details_path: str, benchmark: str
    ) -> Tuple[str, Dict[str, ModelExample], List[str]]:
        """Load evaluation details for a single model file.

        Args:
            model_eval_details_path: Path to the model evaluation details file
            benchmark: Name of the benchmark

        Returns:
            Tuple of (model_name, model_examples_dict, task_ids_list)
        """
        print(f"Loading model eval details: {model_eval_details_path}")
        with open(model_eval_details_path, "r") as f:
            model_eval_details = json.load(f)

        model_name = os.path.basename(model_eval_details_path).replace(
            "_eval_details.json", ""
        )

        model_examples = {}
        task_ids = []

        for task_id, task_eval_details in model_eval_details.items():
            task_ids.append(task_id)

            if benchmark == "ifeval":
                task_eval_details["answer"] = "No ground truth answer."

            model_examples[task_id] = ModelExample(
                correct=task_eval_details["correct"],
                score=task_eval_details["score"],
                problem=task_eval_details["problem"],
                generation=task_eval_details["generation"],
                prediction=task_eval_details["prediction"],
                answer=task_eval_details["answer"],
            )

        return model_name, model_examples, task_ids

    def _get_benchmark_data(
        self,
        max_workers: int = None,
    ):
        """Get the benchmark to model name to task id to eval details.

        Args:
            max_workers: Maximum number of worker threads. If None, uses min(32, os.cpu_count() + 4)

        Returns:
            dict[str, dict[str, dict[str, dict[str, dict]]]]]: A dictionary of dictionaries containing the evaluation details.
            In the format of:
            {
                <benchmark_name>: {
                    BenchmarkData(
                        model_examples: {
                            <model_name>: {
                                <task_id>: ModelExample,
                                ...
                            }, ...
                        },
                        task_ids: [<task_id>, ...],
                    )
                }, ...
        """
        benchmark_data = {}

        # Get the model eval details directory
        model_eval_details_dir = os.path.join(
            self.experiment_path, "eval", "model_eval_details"
        )

        # Set default max_workers if not provided
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        # Process each benchmark
        for benchmark in self.existing_benchmarks:
            print(f"Processing benchmark: {benchmark}")

            # Get the model eval details paths
            model_eval_details_paths = glob.glob(
                os.path.join(model_eval_details_dir, benchmark, "*_eval_details.json")
            )

            if not model_eval_details_paths:
                print(f"No evaluation files found for benchmark: {benchmark}")
                continue

            # Prepare arguments for parallel processing
            load_args = [(path, benchmark) for path in model_eval_details_paths]

            # Use ThreadPoolExecutor for parallel loading
            model_examples = {}
            all_task_ids = set()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(
                        self._load_single_model_eval_details, path, benchmark
                    ): path
                    for path, benchmark in load_args
                }

                # Collect results as they complete
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        model_name, model_examples_dict, task_ids = future.result()
                        model_examples[model_name] = model_examples_dict
                        all_task_ids.update(task_ids)
                    except Exception as exc:
                        print(f"Error loading {path}: {exc}")

            # Create the benchmark data
            benchmark_data[benchmark] = BenchmarkData(
                model_examples=model_examples,
                task_ids=list(all_task_ids),
            )

        return benchmark_data

    def get_model_examples_for_selection_type_task_and_generation(
        self,
        task_id: str,
        generation: int,
        selection_type: str,
        top_N: int,
        benchmark: str,
    ):
        """Get the model examples for a specific task and generation for a given selection type."""

        # Get the top N model names for the given selection type, generation
        top_N_model_names = self.get_top_models_for_selection_type(
            selection_type=selection_type,
            top_N=top_N,
            generation=generation,
        )

        # Get the model examples for the given task and top N model names
        model_examples = {}
        for model_name in top_N_model_names:
            model_examples[model_name] = self.benchmark_data[benchmark].model_examples[
                model_name
            ][task_id]

        return model_examples

    def get_number_of_correct_solutions_per_generation(
        self,
        selection_type: str,
        benchmark: str,
        task_id: str,
        top_N: int,
    ):
        """Get the number of correct solutions for the task per generation.

        Returns:
            list[tuple[int, int]]: A list of tuples containing the generation and the number of correct solutions.
            In the format of:
            [(<generation>, <number_of_correct_solutions>), ...]
        """

        # Get the number of correct solutions for the task per generation
        correct_solutions_per_generation = []
        for generation in self.get_all_available_gens():
            model_examples = (
                self.get_model_examples_for_selection_type_task_and_generation(
                    task_id=task_id,
                    generation=generation,
                    selection_type=selection_type,
                    top_N=top_N,
                    benchmark=benchmark,
                )
            )
            if len(model_examples) != top_N:
                st.warning(
                    f"Number of model examples for generation {generation} is not equal to top N: {top_N}. Got {len(model_examples)}"
                )

            correct_solutions_per_generation.append(
                (
                    generation,
                    sum(
                        model_example.correct
                        for model_example in model_examples.values()
                    )
                    / len(model_examples)
                    * 100,
                )
            )

        return correct_solutions_per_generation

    def _get_metrics_per_generation(
        self,
        selection_type: str,
        benchmark: str,
        top_N: int,
    ):
        """Get the metrics per generation.

        Returns:
            dict[str, list[tuple[int, float]]]: A dictionary of lists of tuples containing the generation and the metric value.
            In the format of:
            {
                "best_of_N_accuracy": [(<generation>, <metric_value>), ...],
                "majority_vote_accuracy": [(<generation>, <metric_value>), ...],
                "unique_contributions_percentage": [(<generation>, <metric_value>), ...],
            }
        """
        path_to_metrics = os.path.join(
            self.experiment_path,
            "eval",
            "best_of_N",
            benchmark,
            selection_type,
            f"results_N{top_N}.json",
        )
        if not os.path.exists(path_to_metrics):
            st.warning(f"No metrics found for {path_to_metrics}")
            return {}

        with open(path_to_metrics, "r") as f:
            metrics_per_generation = json.load(f)

        def get_unique_contributions_relative_to_all_samples(
            gen_results: dict,
        ):
            """Get the unique contributions relative to all samples."""
            all_unique_contributions = sum(gen_results["unique_contributions"].values())
            return (all_unique_contributions / gen_results["num_unique_samples"]) * 100

        relevant_metrics_per_generation = {}

        for gen_num, gen_results in metrics_per_generation.items():

            # Collect the best of N accuracy
            if "best_of_N_accuracy" not in relevant_metrics_per_generation:
                relevant_metrics_per_generation["best_of_N_accuracy"] = []
            relevant_metrics_per_generation["best_of_N_accuracy"].append(
                (gen_num, gen_results["best_of_N_accuracy"])
            )

            # Collect the majority vote accuracy
            if "majority_vote_accuracy" not in relevant_metrics_per_generation:
                relevant_metrics_per_generation["majority_vote_accuracy"] = []
            relevant_metrics_per_generation["majority_vote_accuracy"].append(
                (gen_num, gen_results["majority_vote_accuracy"])
            )

            # Collect the unique contributions percentage
            if "unique_contributions_percentage" not in relevant_metrics_per_generation:
                relevant_metrics_per_generation["unique_contributions_percentage"] = []
            relevant_metrics_per_generation["unique_contributions_percentage"].append(
                (
                    gen_num,
                    get_unique_contributions_relative_to_all_samples(gen_results),
                )
            )

        return relevant_metrics_per_generation

    def get_metrics_for_different_topNs(
        self,
        selection_type: str,
        benchmark: str,
        max_top_N: int,
    ):
        """Get the metrics for different top Ns.

        Returns:
            dict[int, dict[str, list[tuple[int, float]]]]: A dictionary of dictionaries containing the metrics for different top Ns.
            In the format of:
            {
                <top_N>: {
                    "best_of_N_accuracy": [(<generation>, <metric_value>), ...],
                    "majority_vote_accuracy": [(<generation>, <metric_value>), ...],
                    "unique_contributions_percentage": [(<generation>, <metric_value>), ...],
                }, ...
            }
        """
        metrics_per_topN = {}
        if max_top_N < 2:
            max_top_N = 2

        for top_N in range(1, max_top_N + 1):
            metrics_per_generation = self._get_metrics_per_generation(
                selection_type=selection_type,
                benchmark=benchmark,
                top_N=top_N,
            )
            metrics_per_topN[top_N] = metrics_per_generation

        return metrics_per_topN

    def get_metrics_for_model(
        self,
        model_name: str,
        benchmark: str,
    ):
        """Get the metrics for a specific model."""
        path_to_model_eval_details = os.path.join(
            self.experiment_path,
            "eval",
            "model_eval_details",
            benchmark,
            f"{model_name}_metric_details.json",
        )
        if not os.path.exists(path_to_model_eval_details):
            st.warning(f"No model eval details found for {path_to_model_eval_details}")
            return {}

        with open(path_to_model_eval_details, "r") as f:
            model_eval_details = json.load(f)

        if benchmark not in self.benchmark_to_metric_name:
            st.warning(
                f"Benchmark {benchmark} not found in benchmark_to_metric_name. "
                f"Available benchmarks: {self.benchmark_to_metric_name.keys()}"
            )
            return {}

        metric_score = model_eval_details["results"][
            self.benchmark_to_metric_name[benchmark]
        ]

        return metric_score

    def get_mmlu_group_scores_for_model(
        self,
        model_name: str,
        use_llm_as_a_judge: bool = False,
        benchmark_name: str = "mmlu_cot_llama",
    ):
        """Get the MMLU group scores for a specific model."""

        if use_llm_as_a_judge:
            metric_name = self.benchmark_to_metric_name[
                f"{benchmark_name}_llm_as_a_judge"
            ]
        else:
            metric_name = self.benchmark_to_metric_name[benchmark_name]

        path_to_model_eval_details = os.path.join(
            self.experiment_path,
            "eval",
            "model_eval_details",
            (
                f"{benchmark_name}_llm_as_a_judge"
                if use_llm_as_a_judge
                else benchmark_name
            ),
            f"{model_name}_metric_details.json",
        )
        if not os.path.exists(path_to_model_eval_details):
            st.warning(f"No model eval details found for {path_to_model_eval_details}")
            return {}

        with open(path_to_model_eval_details, "r") as f:
            model_eval_details = json.load(f)

        mmlu_group_scores = {}

        for group_name, group_scores in model_eval_details["groups"].items():
            if use_llm_as_a_judge:
                if f"{benchmark_name}_llm_as_a_judge" not in group_name:
                    continue
            else:
                if benchmark_name not in group_name or "llm_as_a_judge" in group_name:
                    continue
            group_name = group_name.replace(" - ", "")

            # if there is content after benchmark_name, remove benchmark_name
            if use_llm_as_a_judge:
                full_benchmark_name = f"{benchmark_name}_llm_as_a_judge"
            else:
                full_benchmark_name = benchmark_name
            if f"{full_benchmark_name}_" in group_name:
                group_name = group_name.split(f"{full_benchmark_name}_")[-1]

            mmlu_group_scores[group_name] = group_scores[metric_name]

        return mmlu_group_scores

    def get_top_models_for_selection_type(
        self,
        selection_type: str,
        generation: int = None,
        top_N: int = 8,
    ):
        """Get the top models for a specific selection type."""
        if selection_type in ["coverage", "fitness"] and generation is not None:
            top_N_model_names = self.gen_to_selection_type_topN_to_model_names[
                generation
            ][selection_type][top_N]
        else:
            # get the best_of_N/<benchmark>/<selection_type>/results_N<top_N>.json
            # for one example benchmark
            benchmark = os.path.basename(
                glob.glob(os.path.join(self.experiment_path, "eval", "best_of_N", "*"))[
                    0
                ]
            )
            path_to_results = os.path.join(
                self.experiment_path,
                "eval",
                "best_of_N",
                benchmark,
                selection_type,
                f"results_N{top_N}.json",
            )
            with open(path_to_results, "r") as f:
                results = json.load(f)
            top_N_model_names = list(results["individual_accuracies"].keys())

        return top_N_model_names


@st.cache_data(ttl=60 * 60)
def create_generation_tree(
    models_to_parent_models: Dict[str, List[Tuple[str, float]]],
):
    """Create the generation tree."""
    # Create a directed graph
    G = nx.DiGraph()

    # Helper function to check if model is a seed model
    def is_seed_model(model_name: str) -> bool:
        """Check if model is a seed model (doesn't have gen_X pattern)."""
        # Check if model name contains "gen_" and a number
        return not re.search(r"gen_\d+", model_name)

    ### Prepare basic nodes and edges ##################################
    # Get all model anmes from parent models
    all_model_names = set()
    for model_name, parent_models in models_to_parent_models.items():
        all_model_names.add(model_name)
        for parent_model, parent_model_fitness in parent_models:
            all_model_names.add(parent_model)

    # Find seed models
    seed_models = set([node for node in all_model_names if is_seed_model(node)])

    # Add all nodes to the graph
    for model_name in all_model_names:
        if model_name in seed_models:
            generation = -1
        else:
            generation = extract_generation(model_name)

        parent_models = models_to_parent_models.get(model_name, [])
        parent_model_names = [
            parent_model.split("/")[-1] for parent_model, _ in parent_models
        ]

        G.add_node(
            model_name,
            generation=generation,
            is_seed=model_name in seed_models,
            parent_models=parent_model_names,
        )

    # Add all other models to the graph
    for model_name, parent_models in models_to_parent_models.items():
        # Add edges from parents to this model
        for parent_model, parent_model_fitness in parent_models:
            if (
                parent_model in models_to_parent_models or parent_model in seed_models
            ):  # Only add if parent exists
                G.add_edge(parent_model, model_name, fitness=parent_model_fitness)

    return G


# ttl (seconds): here 60 * 60 = 1 hour
@st.cache_data(ttl=60 * 60)
def load_experiment_data(experiment_path: str) -> tuple[ExperimentData, dict]:
    """Load experiment data with caching to avoid reloading on every rerun."""
    try:
        task_name_to_tSNE_embedding, task_name_to_hdbscan_cluster = (
            compute_and_load_synth_task_to_tSNE_mapping(experiment_path)
        )
    except Exception as e:
        st.error(f"Error loading tSNE embedding: {e}")
        task_name_to_tSNE_embedding = {}
        task_name_to_hdbscan_cluster = {}
    return (
        ExperimentData(experiment_path),
        task_name_to_tSNE_embedding,
        task_name_to_hdbscan_cluster,
    )


# ttl (seconds): here 60 * 60 = 1 hour
@st.cache_data(ttl=60 * 60)
def load_baselines_results(
    baselines_results_dir_path: str, relevant_benchmarks: list[str]
) -> tuple[dict, dict, dict, dict]:
    """Load baselines results."""
    path_to_best_of_N_results = os.path.join(baselines_results_dir_path, "best_of_N")

    baselines_results_experts = {}
    baselines_results_big_model = {}
    baselines_results_control = {}

    benchmark_dirs = glob.glob(os.path.join(path_to_best_of_N_results, "*"))

    expert_results_exist = False
    big_model_results_exist = False
    control_results_N5_exist = False
    control_results_N8_exist = False
    for benchmark_dir in benchmark_dirs:
        if os.path.basename(benchmark_dir) not in relevant_benchmarks:
            continue

        benchmark_name = os.path.basename(benchmark_dir)

        ### Load the best of N results for the experts
        if os.path.exists(os.path.join(benchmark_dir, "results_N3.json")):
            with open(os.path.join(benchmark_dir, "results_N3.json"), "r") as f:
                expert_results = json.load(f)
            unique_contributions_percentage = (
                get_unique_contributions_relative_to_all_samples(expert_results)
            )
            relevant_metrics = {
                "best_of_N_accuracy": expert_results["best_of_N_accuracy"],
                "majority_vote_accuracy": expert_results["majority_vote_accuracy"],
                "unique_contributions_percentage": unique_contributions_percentage,
            }
            expert_results_exist = True
        else:
            relevant_metrics = {
                "best_of_N_accuracy": 0,
                "majority_vote_accuracy": 0,
                "unique_contributions_percentage": 0,
            }
        baselines_results_experts[benchmark_name] = relevant_metrics

        ### Load the best of N results for the big models
        big_model_results = {}
        big_model_result_files = glob.glob(
            os.path.join(benchmark_dir, "big_model_*_N1.json")
        )
        for big_model_result_file in big_model_result_files:
            with open(big_model_result_file, "r") as f:
                big_model_results_for_one_model = json.load(f)
            model_size_regex = r"(\d+)B"
            model_size_match = re.search(
                model_size_regex, os.path.basename(big_model_result_file)
            )
            if model_size_match:
                model_size = model_size_match.group(1)
            else:
                st.warning(f"No model size found in {big_model_result_file}")
                continue
            big_model_results[model_size] = {
                "best_of_N_accuracy": big_model_results_for_one_model[
                    "best_of_N_accuracy"
                ],
                "majority_vote_accuracy": big_model_results_for_one_model[
                    "majority_vote_accuracy"
                ],
                "unique_contributions_percentage": 0,
            }

        if big_model_results:
            big_model_results_exist = True

        baselines_results_big_model[benchmark_name] = big_model_results

        ### Load the best of N results for the control
        # N = 5
        if os.path.exists(os.path.join(benchmark_dir, "control_results_N5.json")):
            with open(os.path.join(benchmark_dir, "control_results_N5.json"), "r") as f:
                control_results = json.load(f)
            relevant_metrics = {
                "best_of_N_accuracy": control_results["best_of_N_accuracy"],
                "majority_vote_accuracy": control_results["majority_vote_accuracy"],
                "unique_contributions_percentage": 0,
            }
            control_results_N5_exist = True
        else:
            relevant_metrics = {
                "best_of_N_accuracy": 0,
                "majority_vote_accuracy": 0,
                "unique_contributions_percentage": 0,
            }
        if benchmark_name not in baselines_results_control:
            baselines_results_control[benchmark_name] = {}
        baselines_results_control[benchmark_name][5] = relevant_metrics

        # N = 8
        if os.path.exists(os.path.join(benchmark_dir, "control_results_N8.json")):
            with open(os.path.join(benchmark_dir, "control_results_N8.json"), "r") as f:
                control_results = json.load(f)
            relevant_metrics = {
                "best_of_N_accuracy": control_results["best_of_N_accuracy"],
                "majority_vote_accuracy": control_results["majority_vote_accuracy"],
                "unique_contributions_percentage": 0,
            }
            control_results_N8_exist = True
        else:
            relevant_metrics = {
                "best_of_N_accuracy": 0,
                "majority_vote_accuracy": 0,
                "unique_contributions_percentage": 0,
            }
        if benchmark_name not in baselines_results_control:
            baselines_results_control[benchmark_name] = {}
        baselines_results_control[benchmark_name][8] = relevant_metrics

    if not expert_results_exist:
        st.warning(
            "No expert results found for at least one benchmark. "
            "This means that the experts were not evaluated on at least one benchmark."
        )
    if not big_model_results_exist:
        st.warning(
            "No big model results found for at least one benchmark. "
            "This means that the big models were not evaluated on at least one benchmark."
        )
    if not control_results_N5_exist:
        st.warning(
            "No control results found for N=5 for at least one benchmark. "
            "This means that the control was not evaluated on at least one benchmark."
        )
    if not control_results_N8_exist:
        st.warning(
            "No control results found for N=8 for at least one benchmark. "
            "This means that the control was not evaluated on at least one benchmark."
        )

    # Get the single answer from pop data for the baselines
    baselines_single_answer_from_pop_data = get_single_answer_from_pop_data(
        os.path.join(baselines_results_dir_path, "single_answer_from_pop"),
        relevant_benchmarks,
        is_baselines=True,
    )

    return (
        baselines_results_experts,
        baselines_results_big_model,
        baselines_results_control,
        baselines_single_answer_from_pop_data,
    )


def show_metric_evolution_for_one_benchmark(
    experiment_data: ExperimentData,
    selection_type: str,
    benchmark: str,
    baselines_results_experts: dict,
    baselines_results_big_model: dict,
    baselines_results_control: dict,
    max_top_N: int,
):
    """Shows the evolution of metrics for a specific benchmark."""

    st.subheader(f"Benchmark '{benchmark}'")

    # Get the metrics for different top Ns
    metrics_per_topN = experiment_data.get_metrics_for_different_topNs(
        selection_type=selection_type,
        benchmark=benchmark,
        max_top_N=max_top_N,
    )

    # Create one figure for every metric
    # Add one line chart for each top N to each metric
    fig_best_of_N_accuracy = make_subplots(rows=1, cols=1)
    fig_majority_vote_accuracy = make_subplots(rows=1, cols=1)
    fig_unique_contributions_percentage = make_subplots(rows=1, cols=1)

    ### Add the lines for the top Ns ####################################
    ### Archive
    for top_N, metrics_per_generation in metrics_per_topN.items():
        # Add the best of N accuracy line chart
        fig_best_of_N_accuracy.add_scatter(
            x=[gen_num for gen_num, _ in metrics_per_generation["best_of_N_accuracy"]],
            y=[
                metric_value
                for _, metric_value in metrics_per_generation["best_of_N_accuracy"]
            ],
            name=f"N={top_N}",
            line=dict(color=COLOR_MAP[top_N]),
        )

        # Add the majority vote accuracy line chart
        fig_majority_vote_accuracy.add_scatter(
            x=[
                gen_num
                for gen_num, _ in metrics_per_generation["majority_vote_accuracy"]
            ],
            y=[
                metric_value
                for _, metric_value in metrics_per_generation["majority_vote_accuracy"]
            ],
            name=f"N={top_N}",
            line=dict(color=COLOR_MAP[top_N]),
        )

        # Add the unique contributions percentage line chart, skipping N=1
        if top_N > 1:
            fig_unique_contributions_percentage.add_scatter(
                x=[
                    gen_num
                    for gen_num, _ in metrics_per_generation[
                        "unique_contributions_percentage"
                    ]
                ],
                y=[
                    metric_value
                    for _, metric_value in metrics_per_generation[
                        "unique_contributions_percentage"
                    ]
                ],
                name=f"N={top_N}",
                line=dict(color=COLOR_MAP[top_N]),
            )

    ### Add N=1 to the unique contributions legend without showing its line
    first_gen = min(
        metrics_per_topN[1]["unique_contributions_percentage"],
        key=lambda x: x[0],
    )[0]
    fig_unique_contributions_percentage.add_scatter(
        x=[first_gen],
        y=[0],
        name="N=1",
        showlegend=True,
        visible="legendonly",  # This makes the line invisible but keeps it in the legend
        line=dict(color=COLOR_MAP[1]),
    )

    ### Add baselines
    color_map_counter = 9
    if baselines_results_experts:
        if benchmark in baselines_results_experts.keys():
            # x values are all generations
            x_values = [
                gen_num for gen_num, _ in metrics_per_topN[1]["best_of_N_accuracy"]
            ]
            # line details for experts
            line_details_experts = dict(
                color=COLOR_MAP[color_map_counter], width=2, dash="dash"
            )
            color_map_counter += 1
            fig_best_of_N_accuracy.add_scatter(
                x=x_values,
                y=[baselines_results_experts[benchmark]["best_of_N_accuracy"]]
                * len(x_values),
                name="N3 - Experts",
                line=line_details_experts,
            )
            fig_majority_vote_accuracy.add_scatter(
                x=x_values,
                y=[baselines_results_experts[benchmark]["majority_vote_accuracy"]]
                * len(x_values),
                name="N3 - Experts",
                line=line_details_experts,
            )
            fig_unique_contributions_percentage.add_scatter(
                x=x_values,
                y=[
                    baselines_results_experts[benchmark][
                        "unique_contributions_percentage"
                    ]
                ]
                * len(x_values),
                name="N3 - Experts",
                line=line_details_experts,
            )

    ### Add big model baselines
    if baselines_results_big_model:
        if benchmark in baselines_results_big_model.keys():
            for model_size, big_model_results in baselines_results_big_model[
                benchmark
            ].items():
                # line details for big model
                line_details_big_model = dict(
                    color=COLOR_MAP[color_map_counter], width=2, dash="dash"
                )
                color_map_counter += 1
                fig_best_of_N_accuracy.add_scatter(
                    x=x_values,
                    y=[big_model_results["best_of_N_accuracy"]] * len(x_values),
                    name=f"N1 - Big Model {model_size}B",
                    line=line_details_big_model,
                )
                fig_majority_vote_accuracy.add_scatter(
                    x=x_values,
                    y=[big_model_results["majority_vote_accuracy"]] * len(x_values),
                    name=f"N1 - Big Model {model_size}B",
                    line=line_details_big_model,
                )
                fig_unique_contributions_percentage.add_scatter(
                    x=x_values,
                    y=[big_model_results["unique_contributions_percentage"]]
                    * len(x_values),
                    name=f"N1 - Big Model {model_size}B",
                    line=line_details_big_model,
                    visible="legendonly",
                )

    ### Add control baselines
    if baselines_results_control:
        if benchmark in baselines_results_control.keys():
            for top_N in [5, 8]:
                fig_best_of_N_accuracy.add_scatter(
                    x=x_values,
                    y=[
                        baselines_results_control[benchmark][top_N][
                            "best_of_N_accuracy"
                        ]
                    ]
                    * len(x_values),
                    name=f"N{top_N} - Control",
                    line=dict(color=COLOR_MAP[color_map_counter], width=2, dash="dash"),
                )
                fig_majority_vote_accuracy.add_scatter(
                    x=x_values,
                    y=[
                        baselines_results_control[benchmark][top_N][
                            "majority_vote_accuracy"
                        ]
                    ]
                    * len(x_values),
                    name=f"N{top_N} - Control",
                    line=dict(color=COLOR_MAP[color_map_counter], width=2, dash="dash"),
                )
                fig_unique_contributions_percentage.add_scatter(
                    x=x_values,
                    y=[
                        baselines_results_control[benchmark][top_N][
                            "unique_contributions_percentage"
                        ]
                    ]
                    * len(x_values),
                    name=f"N{top_N} - Control",
                    line=dict(color=COLOR_MAP[color_map_counter], width=2, dash="dash"),
                )
                color_map_counter += 1

    ### Add the relevant descriptions to each plot #####################
    fig_best_of_N_accuracy.update_layout(
        title=f"Best of N Accuracy",
        xaxis_title="Generation",
        yaxis_title="Best of N Accuracy (%)",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            ticktext=[
                str(gen_num) for gen_num, _ in metrics_per_topN[1]["best_of_N_accuracy"]
            ],
            tickvals=[
                gen_num for gen_num, _ in metrics_per_topN[1]["best_of_N_accuracy"]
            ],
        ),
    )

    fig_majority_vote_accuracy.update_layout(
        title=f"Majority Vote Accuracy",
        xaxis_title="Generation",
        yaxis_title="Majority Vote Accuracy (%)",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            ticktext=[
                str(gen_num)
                for gen_num, _ in metrics_per_topN[1]["majority_vote_accuracy"]
            ],
            tickvals=[
                gen_num for gen_num, _ in metrics_per_topN[1]["majority_vote_accuracy"]
            ],
        ),
    )

    # Get the first available N > 1 for unique contributions plot
    first_n_gt_1 = next(n for n in metrics_per_topN.keys() if n > 1)
    fig_unique_contributions_percentage.update_layout(
        title=f"Unique Contributions (%)",
        xaxis_title="Generation",
        yaxis_title="Unique Contributions (%)",
        showlegend=True,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
        ),
        xaxis=dict(
            tickmode="array",
            ticktext=[
                str(gen_num)
                for gen_num, _ in metrics_per_topN[first_n_gt_1][
                    "unique_contributions_percentage"
                ]
            ],
            tickvals=[
                gen_num
                for gen_num, _ in metrics_per_topN[first_n_gt_1][
                    "unique_contributions_percentage"
                ]
            ],
        ),
    )

    ### Show the figures next to each other ############################
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_best_of_N_accuracy)
    with col2:
        st.plotly_chart(fig_majority_vote_accuracy)
    with col3:
        st.plotly_chart(fig_unique_contributions_percentage)


def show_metric_evolution_per_generation(
    experiment_data: ExperimentData,
    selection_type: str,
    baselines_results_experts: dict,
    baselines_results_big_model: dict,
    baselines_results_control: dict,
    top_N: int,
):
    """Shows a line chart of all relevant metrics per generation.

    The metrics are:
    - best of N score
    - majroity vote score
    - unique contributions (%)
    """

    show_average_metric_evolution(
        experiment_data=experiment_data,
        selection_type=selection_type,
        baselines_results_experts=baselines_results_experts,
        baselines_results_big_model=baselines_results_big_model,
        baselines_results_control=baselines_results_control,
        top_N=top_N,
    )

    # get plots for all benchmarks
    for benchmark in experiment_data.existing_benchmarks:
        show_metric_evolution_for_one_benchmark(
            experiment_data=experiment_data,
            selection_type=selection_type,
            benchmark=benchmark,
            baselines_results_experts=baselines_results_experts,
            baselines_results_big_model=baselines_results_big_model,
            baselines_results_control=baselines_results_control,
            max_top_N=top_N,
        )


def show_average_metric_evolution(
    experiment_data: ExperimentData,
    selection_type: str,
    baselines_results_experts: dict,
    baselines_results_big_model: dict,
    baselines_results_control: dict,
    top_N: int,
):
    """Shows a line chart of average metrics across all benchmarks per generation."""

    st.subheader(
        f"Average cross all {len(experiment_data.existing_benchmarks)} benchmarks"
    )
    if top_N < 2:
        top_N = 2

    # Get metrics for all benchmarks
    all_metrics = {}
    for benchmark in experiment_data.existing_benchmarks:
        metrics = experiment_data.get_metrics_for_different_topNs(
            selection_type=selection_type,
            benchmark=benchmark,
            max_top_N=top_N,
        )
        all_metrics[benchmark] = metrics

    # Calculate average metrics
    average_metrics = {}
    for n in range(1, top_N + 1):
        average_metrics[n] = {
            "best_of_N_accuracy": [],
            "majority_vote_accuracy": [],
            "unique_contributions_percentage": [],
        }

        # Get all generations from first benchmark
        first_benchmark = experiment_data.existing_benchmarks[0]
        generations = [
            gen_num
            for gen_num, _ in all_metrics[first_benchmark][n]["best_of_N_accuracy"]
        ]

        # For each generation
        for gen_num in generations:
            sums = {
                "best_of_N_accuracy": 0.0,
                "majority_vote_accuracy": 0.0,
                "unique_contributions_percentage": 0.0,
            }
            valid_benchmarks = 0

            # For each benchmark
            for benchmark in experiment_data.existing_benchmarks:
                metrics = all_metrics[benchmark][n]

                # Find the metric value for this generation
                for metric_name in sums.keys():
                    for g, value in metrics[metric_name]:
                        if g == gen_num:
                            sums[metric_name] += value
                            break
                valid_benchmarks += 1

            # Calculate averages for this generation
            if valid_benchmarks > 0:
                for metric_name in sums.keys():
                    avg_value = sums[metric_name] / valid_benchmarks
                    average_metrics[n][metric_name].append((gen_num, avg_value))

    # Create one figure for every metric
    fig_best_of_N_accuracy = make_subplots(rows=1, cols=1)
    fig_majority_vote_accuracy = make_subplots(rows=1, cols=1)
    fig_unique_contributions_percentage = make_subplots(rows=1, cols=1)

    ### Add the lines for the top Ns
    for n in range(1, top_N + 1):
        # Add the best of N accuracy line chart
        fig_best_of_N_accuracy.add_scatter(
            x=[gen_num for gen_num, _ in average_metrics[n]["best_of_N_accuracy"]],
            y=[
                metric_value
                for _, metric_value in average_metrics[n]["best_of_N_accuracy"]
            ],
            name=f"N={n}",
            line=dict(color=COLOR_MAP[n]),
        )

        # Add the majority vote accuracy line chart
        fig_majority_vote_accuracy.add_scatter(
            x=[gen_num for gen_num, _ in average_metrics[n]["majority_vote_accuracy"]],
            y=[
                metric_value
                for _, metric_value in average_metrics[n]["majority_vote_accuracy"]
            ],
            name=f"N={n}",
            line=dict(color=COLOR_MAP[n]),
        )

        # Add the unique contributions percentage line chart, skipping N=1
        if n > 1:
            fig_unique_contributions_percentage.add_scatter(
                x=[
                    gen_num
                    for gen_num, _ in average_metrics[n][
                        "unique_contributions_percentage"
                    ]
                ],
                y=[
                    metric_value
                    for _, metric_value in average_metrics[n][
                        "unique_contributions_percentage"
                    ]
                ],
                name=f"N={n}",
                line=dict(color=COLOR_MAP[n]),
            )

    ### Add N=1 to the unique contributions legend without showing its line
    first_gen = min(
        average_metrics[1]["unique_contributions_percentage"],
        key=lambda x: x[0],
    )[0]
    fig_unique_contributions_percentage.add_scatter(
        x=[first_gen],
        y=[0],
        name="N=1",
        showlegend=True,
        visible="legendonly",
        line=dict(color=COLOR_MAP[1]),
    )

    ### Add baselines
    color_map_counter = 9
    # x values are all generations
    x_values = [gen_num for gen_num, _ in average_metrics[1]["best_of_N_accuracy"]]
    # Calculate average baseline values
    # Experts
    if baselines_results_experts:
        sum_experts_best_of_N = 0
        n_benchmarks_w_expert_results = 0
        sum_experts_majority = 0
        sum_experts_unique = 0
        for benchmark in experiment_data.existing_benchmarks:
            if benchmark in baselines_results_experts.keys():
                sum_experts_best_of_N += baselines_results_experts[benchmark][
                    "best_of_N_accuracy"
                ]
                sum_experts_majority += baselines_results_experts[benchmark][
                    "majority_vote_accuracy"
                ]
                sum_experts_unique += baselines_results_experts[benchmark][
                    "unique_contributions_percentage"
                ]
                n_benchmarks_w_expert_results += 1
            else:
                st.warning(f"No experts results for benchmark {benchmark}")
        avg_expert_best_of_N = sum_experts_best_of_N / n_benchmarks_w_expert_results
        avg_expert_majority = sum_experts_majority / n_benchmarks_w_expert_results
        avg_expert_unique = sum_experts_unique / n_benchmarks_w_expert_results

        line_details_experts = dict(
            color=COLOR_MAP[color_map_counter], width=2, dash="dash"
        )
        color_map_counter += 1

        fig_best_of_N_accuracy.add_scatter(
            x=x_values,
            y=[avg_expert_best_of_N] * len(x_values),
            name="N3 - Experts",
            line=line_details_experts,
        )
        fig_majority_vote_accuracy.add_scatter(
            x=x_values,
            y=[avg_expert_majority] * len(x_values),
            name="N3 - Experts",
            line=line_details_experts,
        )
        fig_unique_contributions_percentage.add_scatter(
            x=x_values,
            y=[avg_expert_unique] * len(x_values),
            name="N3 - Experts",
            line=line_details_experts,
        )

    if baselines_results_big_model:
        # Big Model
        model_sizes = list(
            baselines_results_big_model[experiment_data.existing_benchmarks[0]].keys()
        )
        for model_size in model_sizes:
            sum_big_model_best_of_N = 0
            n_benchmarks_w_big_model_results = 0
            sum_big_model_majority = 0
            for benchmark in experiment_data.existing_benchmarks:
                if benchmark in baselines_results_big_model.keys():
                    sum_big_model_best_of_N += baselines_results_big_model[benchmark][
                        model_size
                    ]["best_of_N_accuracy"]
                    sum_big_model_majority += baselines_results_big_model[benchmark][
                        model_size
                    ]["majority_vote_accuracy"]
                    n_benchmarks_w_big_model_results += 1
                else:
                    st.warning(f"No big model results for benchmark {benchmark}")
            avg_big_model_best_of_N = (
                sum_big_model_best_of_N / n_benchmarks_w_big_model_results
            )
            avg_big_model_majority = (
                sum_big_model_majority / n_benchmarks_w_big_model_results
            )

            ### Add big model baselines
            line_details_big_model = dict(
                color=COLOR_MAP[color_map_counter], width=2, dash="dash"
            )
            color_map_counter += 1
            fig_best_of_N_accuracy.add_scatter(
                x=x_values,
                y=[avg_big_model_best_of_N] * len(x_values),
                name=f"N1 - Big Model {model_size}B",
                line=line_details_big_model,
            )
            fig_majority_vote_accuracy.add_scatter(
                x=x_values,
                y=[avg_big_model_majority] * len(x_values),
                name=f"N1 - Big Model {model_size}B",
                line=line_details_big_model,
            )
            fig_unique_contributions_percentage.add_scatter(
                x=x_values,
                y=[0] * len(x_values),
                name=f"N1 - Big Model {model_size}B",
                line=line_details_big_model,
                visible="legendonly",
            )

    if baselines_results_control:
        ### Control
        # N=5
        sum_control_best_of_N5 = 0
        n_benchmarks_w_control_results = 0
        sum_control_majority5 = 0
        for benchmark in experiment_data.existing_benchmarks:
            if benchmark in baselines_results_control.keys():
                sum_control_best_of_N5 += baselines_results_control[benchmark][5][
                    "best_of_N_accuracy"
                ]
                sum_control_majority5 += baselines_results_control[benchmark][5][
                    "majority_vote_accuracy"
                ]
                n_benchmarks_w_control_results += 1
            else:
                st.warning(f"No control N=5 results for benchmark {benchmark}")
        avg_control_best_of_N5 = sum_control_best_of_N5 / n_benchmarks_w_control_results
        avg_control_majority5 = sum_control_majority5 / n_benchmarks_w_control_results

        # N=8
        sum_control_best_of_N8 = 0
        n_benchmarks_w_control_results = 0
        sum_control_majority8 = 0
        for benchmark in experiment_data.existing_benchmarks:
            if benchmark in baselines_results_control.keys():
                sum_control_best_of_N8 += baselines_results_control[benchmark][8][
                    "best_of_N_accuracy"
                ]
                sum_control_majority8 += baselines_results_control[benchmark][8][
                    "majority_vote_accuracy"
                ]
                n_benchmarks_w_control_results += 1
            else:
                st.warning(f"No control N=8 results for benchmark {benchmark}")
        avg_control_best_of_N8 = sum_control_best_of_N8 / n_benchmarks_w_control_results

        avg_control_majority8 = sum_control_majority8 / n_benchmarks_w_control_results

        ### Add control baselines
        line_details_control_N5 = dict(
            color=COLOR_MAP[color_map_counter], width=2, dash="dash"
        )
        color_map_counter += 1
        line_details_control_N8 = dict(
            color=COLOR_MAP[color_map_counter], width=2, dash="dash"
        )
        color_map_counter += 1
        fig_best_of_N_accuracy.add_scatter(
            x=x_values,
            y=[avg_control_best_of_N5] * len(x_values),
            name="N5 - Control",
            line=line_details_control_N5,
        )
        fig_majority_vote_accuracy.add_scatter(
            x=x_values,
            y=[avg_control_majority5] * len(x_values),
            name="N5 - Control",
            line=line_details_control_N5,
        )
        fig_unique_contributions_percentage.add_scatter(
            x=x_values,
            y=[0] * len(x_values),
            name="N5 - Control",
            line=line_details_control_N5,
            visible="legendonly",
        )
        fig_best_of_N_accuracy.add_scatter(
            x=x_values,
            y=[avg_control_best_of_N8] * len(x_values),
            name="N8 - Control",
            line=line_details_control_N8,
        )
        fig_majority_vote_accuracy.add_scatter(
            x=x_values,
            y=[avg_control_majority8] * len(x_values),
            name="N8 - Control",
            line=line_details_control_N8,
        )
        fig_unique_contributions_percentage.add_scatter(
            x=x_values,
            y=[0] * len(x_values),
            name="N8 - Control",
            line=line_details_control_N8,
            visible="legendonly",
        )

    ### Add the relevant descriptions to each plot
    fig_best_of_N_accuracy.update_layout(
        title=f"Average Best of N Accuracy",
        xaxis_title="Generation",
        yaxis_title="Best of N Accuracy (%)",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            ticktext=[
                str(gen_num) for gen_num, _ in average_metrics[1]["best_of_N_accuracy"]
            ],
            tickvals=[
                gen_num for gen_num, _ in average_metrics[1]["best_of_N_accuracy"]
            ],
        ),
    )

    fig_majority_vote_accuracy.update_layout(
        title=f"Average Majority Vote Accuracy",
        xaxis_title="Generation",
        yaxis_title="Majority Vote Accuracy (%)",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            ticktext=[
                str(gen_num)
                for gen_num, _ in average_metrics[1]["majority_vote_accuracy"]
            ],
            tickvals=[
                gen_num for gen_num, _ in average_metrics[1]["majority_vote_accuracy"]
            ],
        ),
    )

    # Get the first available N > 1 for unique contributions plot
    first_n_gt_1 = next(n for n in average_metrics.keys() if n > 1)
    fig_unique_contributions_percentage.update_layout(
        title=f"Average Unique Contributions (%)",
        xaxis_title="Generation",
        yaxis_title="Unique Contributions (%)",
        showlegend=True,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
        ),
        xaxis=dict(
            tickmode="array",
            ticktext=[
                str(gen_num)
                for gen_num, _ in average_metrics[first_n_gt_1][
                    "unique_contributions_percentage"
                ]
            ],
            tickvals=[
                gen_num
                for gen_num, _ in average_metrics[first_n_gt_1][
                    "unique_contributions_percentage"
                ]
            ],
        ),
    )

    ### Show the figures next to each other
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_best_of_N_accuracy)
    with col2:
        st.plotly_chart(fig_majority_vote_accuracy)
    with col3:
        st.plotly_chart(fig_unique_contributions_percentage)


def show_evolution_chart(
    correct_solutions: List[Tuple[int, int]],
    top_N: int,
):
    """Shows a bar chart of correct solutions per generation as a percentage."""

    df = pd.DataFrame(correct_solutions, columns=["Generation", "Correct Solutions"])

    fig = px.bar(
        df,
        x="Generation",
        y="Correct Solutions",
        title=f"Percentage of Correct Solutions per Generation over Top {top_N} Models.",
        labels={
            "Generation": "Generation",
            "Correct Solutions": "Percentage of Correct Solutions (%)",
        },
    )

    fig.add_scatter(
        x=df["Generation"],
        y=df["Correct Solutions"],
        mode="lines+markers",
        name="Trend",
        line=dict(color="red", width=2),
        marker=dict(size=8),
    )

    gen_step_size = df["Generation"].iloc[-1] - df["Generation"].iloc[-2]

    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=min(df["Generation"]), dtick=gen_step_size),
        yaxis=dict(
            range=[0, 110],
            ticksuffix="%",
        ),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig)


def create_best_of_N_plot(
    best_of_N_results: dict,
    baselines_results: dict,
    benchmark: str,
    selection_method_to_short_name: dict,
) -> go.Figure:
    """Creates a figure showing the best-of-N accuracy for a given benchmark.

    Args:
        best_of_N_results: Dictionary containing the best-of-N results for each selection method
        baselines_results: Dictionary containing the baseline results
        benchmark: Name of the benchmark

    Returns:
        go.Figure: A Plotly figure object
    """
    # Create the main figure
    fig = go.Figure()

    # Plot the best-of-N results per selection method
    for i, (
        selection_method,
        results,
    ) in enumerate(best_of_N_results.items()):
        N_values = sorted(results.keys())
        accuracies = [results[N]["best_of_N_accuracy"] for N in N_values]

        fig.add_trace(
            go.Scatter(
                x=N_values,
                y=accuracies,
                mode="lines+markers",
                name=selection_method_to_short_name[selection_method],
                line=dict(color=COLOR_MAP[i + 1]),
                marker=dict(size=8),
            )
        )

    # Plot the baselines
    # Plot big models if available
    if "big_model" in baselines_results and baselines_results["big_model"]:
        for i, (model_size, big_model_results) in enumerate(
            baselines_results["big_model"].items()
        ):
            fig.add_hline(
                y=big_model_results["best_of_N_accuracy"],
                line_dash="dash",
                line_color=COLOR_MAP[i + 1],
                name=f"Big Model {model_size}B",
                showlegend=True,
            )

    # Plot 3_experts if available
    if "3_experts" in baselines_results and baselines_results["3_experts"]:
        fig.add_hline(
            y=baselines_results["3_experts"]["best_of_N_accuracy"],
            line_dash="dot",
            line_color="gray",
            name="3 Experts",
            showlegend=True,
        )

    # Plot control results if available
    if "control" in baselines_results and baselines_results["control"]:
        control_N_values = sorted(baselines_results["control"].keys())
        control_accuracies = [
            baselines_results["control"][N]["best_of_N_accuracy"]
            for N in control_N_values
        ]

        control_stds = [
            baselines_results["control"][N].get("best_of_N_accuracy_std", 0.0)
            for N in control_N_values
        ]

        if any(std is not None and std != 0 for std in control_stds):
            # Replace None with 0 for stds
            control_stds = [std if std is not None else 0 for std in control_stds]
            upper = [y + s for y, s in zip(control_accuracies, control_stds)]
            lower = [y - s for y, s in zip(control_accuracies, control_stds)]
            fig.add_trace(
                go.Scatter(
                    x=control_N_values + control_N_values[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor="rgba(128,128,128,0.2)",  # semi-transparent gray
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Control Std",
                    legendgroup="Control",
                ),
            )

        fig.add_trace(
            go.Scatter(
                x=control_N_values,
                y=control_accuracies,
                mode="lines+markers",
                name="Control",
                line=dict(color="gray", dash="dash"),
                marker=dict(symbol="square", size=8),
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Best-of-N Results for {benchmark}",
        xaxis_title="Number of Models (N)",
        # yaxis_title="Best-of-N Accuracy (%)",
        # yaxis_title="pass@kModels Accuracy (%)",
        yaxis_title="Coverage",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        # grid=dict(visible=True, color="rgba(128, 128, 128, 0.2)", dash="dash"),
        margin=dict(r=150),  # Add margin for legend
    )

    return fig


# ttl (seconds): here 60 * 60 = 1 hour
@st.cache_data(ttl=60 * 60)
def get_best_of_N_results_for_main_best_of_N_plots(
    experiment_path: str,
    baselines_results_dir_path: str = "visualization/baselines/zero_shot",
    relevant_selection_methods: list[str] = None,
):

    best_of_N_results_dir_path = os.path.join(experiment_path, "eval", "best_of_N")
    baselines_results_dir_path = os.path.join(baselines_results_dir_path, "best_of_N")

    # Get the best-of-N results for each benchmark and selection method
    (
        best_of_N_results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results,
    ) = get_best_of_N_results_per_benchmark_per_selection_method(
        best_of_N_results_dir_path,
        relevant_selection_methods,
    )

    relevant_benchmarks = list(
        best_of_N_results_per_benchmark_per_selection_method.keys()
    )

    # Get the baselines results
    (
        baselines_results_per_benchmark,
        baselines_averaged_across_benchmarks_results,
    ) = get_baselines_results(baselines_results_dir_path, relevant_benchmarks)

    return (
        best_of_N_results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results,
        baselines_results_per_benchmark,
        baselines_averaged_across_benchmarks_results,
    )


def create_best_of_N_subplots_for_benchmarks(
    best_of_N_results_per_benchmark_per_selection_method: dict,
    baselines_results_per_benchmark: dict,
    benchmarks: list[str],
    selection_method_to_short_name: dict,
):
    """Creates a figure with subplots for the given benchmarks."""
    # Create a figure with 3 subplots
    fig = make_subplots(
        rows=1,
        cols=len(benchmarks),
        subplot_titles=benchmarks,
        shared_yaxes=False,
        horizontal_spacing=0.05,
    )

    # Plot the best-of-N results per selection method for each benchmark
    for col_idx, benchmark in enumerate(benchmarks, 1):
        for i, (selection_method, results) in enumerate(
            best_of_N_results_per_benchmark_per_selection_method[benchmark].items()
        ):
            N_values = sorted(results.keys())
            accuracies = [results[N]["best_of_N_accuracy"] for N in N_values]

            # Create the trace
            trace = go.Scatter(
                x=N_values,
                y=accuracies,
                mode="lines+markers",
                name=selection_method_to_short_name[selection_method],
                line=dict(color=COLOR_MAP[i + 1]),
                marker=dict(size=8),
                showlegend=(col_idx == 1),  # Only show legend for first subplot
            )

            # Set legendgroup to group traces with same name across subplots
            trace.legendgroup = selection_method_to_short_name[selection_method]

            # Add trace to the subplot
            fig.add_trace(trace, row=1, col=col_idx)

        # Get the full range of N values for this benchmark to extend horizontal lines
        all_N_values = []
        for (
            selection_method,
            results,
        ) in best_of_N_results_per_benchmark_per_selection_method[benchmark].items():
            all_N_values.extend(sorted(results.keys()))
        max_N = max(all_N_values) if all_N_values else 8

        # Add baselines for this benchmark
        # Plot big models if available
        if (
            benchmark in baselines_results_per_benchmark.keys()
            and "big_model" in baselines_results_per_benchmark[benchmark]
            and baselines_results_per_benchmark[benchmark]["big_model"]
        ):
            big_model_dict = baselines_results_per_benchmark[benchmark]["big_model"]
            for i, (model_size, big_model_results) in enumerate(big_model_dict.items()):
                hline = go.layout.Shape(
                    type="line",
                    x0=1,
                    x1=max_N,
                    y0=big_model_results["best_of_N_accuracy"],
                    y1=big_model_results["best_of_N_accuracy"],
                    xref=f"x{col_idx}",
                    yref=f"y{col_idx}",
                    line=dict(dash="dash", color=COLOR_MAP[i + 1]),
                )
                fig.add_shape(hline)

                # Add legend entry for big model (only for first subplot)
                if col_idx == 1:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=f"Big Model {model_size}B",
                            line=dict(dash="dash", color=COLOR_MAP[i + 1]),
                            showlegend=True,
                            legendgroup=f"Big Model {model_size}B",
                        )
                    )

        # Plot 3_experts if available
        if (
            benchmark in baselines_results_per_benchmark.keys()
            and "3_experts" in baselines_results_per_benchmark[benchmark]
            and baselines_results_per_benchmark[benchmark]["3_experts"]
        ):
            hline = go.layout.Shape(
                type="line",
                x0=1,
                x1=max_N,
                y0=baselines_results_per_benchmark[benchmark]["3_experts"][
                    "best_of_N_accuracy"
                ],
                y1=baselines_results_per_benchmark[benchmark]["3_experts"][
                    "best_of_N_accuracy"
                ],
                xref=f"x{col_idx}",
                yref=f"y{col_idx}",
                line=dict(dash="dot", color="gray"),
            )
            fig.add_shape(hline)

            # Add legend entry for 3 experts (only for first subplot)
            if col_idx == 1:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name="3 Experts",
                        line=dict(dash="dot", color="gray"),
                        showlegend=True,
                        legendgroup="3 Experts",
                    )
                )

        # Plot control results if available
        if (
            benchmark in baselines_results_per_benchmark.keys()
            and "control" in baselines_results_per_benchmark[benchmark]
            and baselines_results_per_benchmark[benchmark]["control"]
        ):
            control_N_values = sorted(
                baselines_results_per_benchmark[benchmark]["control"].keys()
            )
            control_accuracies = [
                baselines_results_per_benchmark[benchmark]["control"][N][
                    "best_of_N_accuracy"
                ]
                for N in control_N_values
            ]
            control_stds = [
                baselines_results_per_benchmark[benchmark]["control"][N].get(
                    "best_of_N_accuracy_std", 0.0
                )
                for N in control_N_values
            ]

            # If available, add "confidence band" of 1 std
            if any(std is not None and std != 0 for std in control_stds):
                # Replace None with 0 for stds
                control_stds = [std if std is not None else 0 for std in control_stds]
                upper = [y + s for y, s in zip(control_accuracies, control_stds)]
                lower = [y - s for y, s in zip(control_accuracies, control_stds)]

                # Add the band (shaded area)
                fig.add_trace(
                    go.Scatter(
                        x=control_N_values + control_N_values[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor="rgba(128,128,128,0.2)",  # semi-transparent gray
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        name="Control Std",
                        legendgroup="Control",
                    ),
                    row=1,
                    col=col_idx,
                )

            control_trace = go.Scatter(
                x=control_N_values,
                y=control_accuracies,
                mode="lines+markers",
                name="Control",
                line=dict(color="gray", dash="dash"),
                marker=dict(symbol="square", size=8),
                showlegend=(col_idx == 1),
                legendgroup="Control",
            )

            # Set legendgroup to group traces with same name across subplots
            control_trace.legendgroup = "Control"

            fig.add_trace(control_trace, row=1, col=col_idx)

    # Update layout
    fig.update_layout(
        height=500,
        width=1500,
        showlegend=True,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
        ),
        margin=dict(r=150),  # Add margin for legend
    )

    # Update y-axis labels
    # fig.update_yaxes(title_text="Best-of-N Accuracy (%)", row=1, col=1)
    # fig.update_yaxes(title_text="pass@kModels Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Coverage", row=1, col=1)

    # Update x-axis labels - only show for middle subplot
    fig.update_xaxes(
        title_text="Number of Models (N)", row=1, col=len(benchmarks) // 2 + 1
    )
    # Hide x-axis labels for other subplots
    for i in range(1, len(benchmarks) + 1):
        if i != len(benchmarks) // 2 + 1:
            fig.update_xaxes(title_text="", row=1, col=i)

    st.plotly_chart(fig, use_container_width=True)


def show_best_of_N_plots(
    best_of_N_results_per_benchmark_per_selection_method: dict,
    averaged_across_benchmarks_results: dict,
    baselines_results_per_benchmark: dict,
    baselines_averaged_across_benchmarks_results: dict,
):
    """Shows the best-of-N accuracy plots for each benchmark.

    Args:
        experiment_path: Path to the experiment directory
        baselines_results_dir_path: Path to the baselines results directory
        relevant_selection_methods: List of selection methods to include in the plots
    """

    st.header("Best-of-N Accuracy Plots")

    # Get the selection methods from the first benchmark
    selection_methods = list(
        best_of_N_results_per_benchmark_per_selection_method[
            list(best_of_N_results_per_benchmark_per_selection_method.keys())[0]
        ].keys()
    )
    # Map selection method to shortened name
    selection_method_to_short_name = {
        selection_method: multi_word_string_to_readable_string(selection_method)
        for selection_method in selection_methods
    }

    st.subheader("Averaged Across Benchmarks Results")
    # Create plot for the averaged across benchmarks results
    fig = create_best_of_N_plot(
        best_of_N_results=averaged_across_benchmarks_results,
        baselines_results=baselines_averaged_across_benchmarks_results,
        benchmark="average over all benchmarks",
        selection_method_to_short_name=selection_method_to_short_name,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best-of-N Accuracy Plots for Each Benchmark")

    # get group of 3 benchmarks
    # sort best_of_N_results_per_benchmark_per_selection_method by name
    sorted_benchmarks = sorted(
        best_of_N_results_per_benchmark_per_selection_method,
        key=lambda x: x.lower(),
    )
    benchmarks_grouped_by_3 = []
    i = 0
    while i < len(sorted_benchmarks):
        end_idx = min(
            i + 3,
            len(sorted_benchmarks),
        )

        benchmarks_grouped_by_3.append(sorted_benchmarks[i:end_idx])
        i += 3

    for benchmarks in benchmarks_grouped_by_3:
        create_best_of_N_subplots_for_benchmarks(
            best_of_N_results_per_benchmark_per_selection_method=best_of_N_results_per_benchmark_per_selection_method,
            baselines_results_per_benchmark=baselines_results_per_benchmark,
            benchmarks=benchmarks,
            selection_method_to_short_name=selection_method_to_short_name,
        )


def show_and_compute_average_metric_for_selected_benchmarks(
    best_of_N_results_dir_path: str,
    baselines_results_dir_path: str,
    relevant_selection_methods: list[str],
    relevant_benchmarks: list[str],
):
    """Compute the average metric for the selected benchmarks."""
    # Get the best of N results for the selected benchmark per selection method
    (
        results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results,
    ) = get_best_of_N_results_per_benchmark_per_selection_method(
        best_of_N_results_dir_path=best_of_N_results_dir_path,
        relevant_selection_methods=relevant_selection_methods,
        relevant_benchmarks=relevant_benchmarks,
    )

    # Get the baselines results
    baselines_results_dir_path = os.path.join(baselines_results_dir_path, "best_of_N")
    (
        baselines_results_per_benchmark,
        baselines_averaged_across_benchmarks_results,
    ) = get_baselines_results(
        baselines_results_dir_path=baselines_results_dir_path,
        relevant_benchmarks=relevant_benchmarks,
    )

    benchmarks_string = ", ".join(relevant_benchmarks)
    st.write(f"Average metric for selected benchmarks: {benchmarks_string}")

    # Get the selection methods from the first benchmark
    selection_methods = list(
        results_per_benchmark_per_selection_method[
            list(results_per_benchmark_per_selection_method.keys())[0]
        ].keys()
    )
    # Map selection method to shortened name
    selection_method_to_short_name = {
        selection_method: multi_word_string_to_readable_string(selection_method)
        for selection_method in selection_methods
    }

    # Create plot for the averaged across benchmarks results
    fig = create_best_of_N_plot(
        best_of_N_results=averaged_across_benchmarks_results,
        baselines_results=baselines_averaged_across_benchmarks_results,
        benchmark=f"average over {benchmarks_string}",
        selection_method_to_short_name=selection_method_to_short_name,
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_answers(
    model_1_examples: Dict[str, ModelExample],
    task_id: str,
    model_2_examples: Dict[str, ModelExample] = None,
    generation1: int = None,
    generation2: int = None,
):
    """Shows the answers of all models for a specific task and generation side by side."""
    # Sort the models by score
    sorted_models1 = sorted(
        model_1_examples.items(),
        key=lambda x: x[1].score,
        reverse=False,
    )

    if model_2_examples:
        sorted_models2 = sorted(
            model_2_examples.items(),
            key=lambda x: x[1].score,
            reverse=False,
        )

    # Get the first model's example to show the task instruction
    first_model_example = next(iter(model_1_examples.values()))
    st.write("**Task Instruction:**")
    st.write(first_model_example.problem)

    st.write("**Model Predictions and Ground Truth:**")

    # Create two columns for side by side comparison
    if model_2_examples:
        st.subheader(
            f"Model Answers in Generations {generation1} and {generation2} for Task {task_id}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Generation {generation1}")
            for model_name, example in sorted_models1:
                with st.expander(
                    f"Model: {model_name} (Score: {example.score:.2f})",
                    expanded=True,
                ):
                    st.write("**Raw Output:**")
                    st.text_area(
                        "",
                        example.generation,
                        height=200,
                        disabled=True,
                        key=f"raw_output_{model_name}_gen{generation1}",
                    )
                    st.write("**Prediction:**")
                    prediction = (
                        example.prediction
                        if example.prediction
                        else "Answer extraction failed"
                    )
                    st.write(prediction)
                    st.write("**Ground Truth:**")
                    st.write(example.answer)
                    st.write("**Correct:**", "✅" if example.correct else "❌")

        with col2:
            st.subheader(f"Generation {generation2}")
            for model_name, example in sorted_models2:
                with st.expander(
                    f"Model: {model_name} (Score: {example.score:.2f})",
                    expanded=True,
                ):
                    st.write("**Raw Output:**")
                    st.text_area(
                        "",
                        example.generation,
                        height=200,
                        disabled=True,
                        key=f"raw_output_{model_name}_gen{generation2}",
                    )
                    st.write("**Prediction:**")
                    prediction = (
                        example.prediction
                        if example.prediction
                        else "Answer extraction failed"
                    )
                    st.write(prediction)
                    st.write("**Ground Truth:**")
                    st.write(example.answer)
                    st.write("**Correct:**", "✅" if example.correct else "❌")

    else:
        # Show model answers for the first generation
        for model_name, example in sorted_models1:
            with st.expander(
                f"Model: {model_name} (Score: {example.score:.2f})",
                expanded=True,
            ):
                st.write("**Raw Output:**")
                st.text_area(
                    "",
                    example.generation,
                    height=200,
                    disabled=True,
                    key=f"raw_output_{model_name}_gen{generation1}",
                )
                st.write("**Prediction:**")
                prediction = (
                    example.prediction
                    if example.prediction
                    else "Answer extraction failed"
                )
                st.write(prediction)
                st.write("**Ground Truth:**")
                st.write(example.answer)
                st.write("**Correct:**", "✅" if example.correct else "❌")


def show_spider_plots_per_model(
    experiment_data: ExperimentData,
    selection_type: str,
    generation: int,
    top_N: int,
):

    st.subheader(
        f"Spider Plots for Top {top_N} Models in Generation {generation} on all Benchmarks"
    )

    # Get the selected models for the selected generation
    top_N_model_names = experiment_data.get_top_models_for_selection_type(
        selection_type, generation, top_N
    )

    # Get the benchmark scores for the selected models
    model_to_benchmark_scores = {model_name: {} for model_name in top_N_model_names}

    for benchmark in experiment_data.existing_benchmarks:
        for model_name in top_N_model_names:
            model_to_benchmark_scores[model_name][benchmark] = (
                experiment_data.get_metrics_for_model(model_name, benchmark)
            )

    # Create a spider plot for each model where each models spider plot
    # is overlayed on top of the other models spider plots
    fig = go.Figure()
    for model_name, benchmark_scores in model_to_benchmark_scores.items():
        fig.add_trace(
            go.Scatterpolar(
                r=list(benchmark_scores.values()),
                theta=list(benchmark_scores.keys()),
                name=model_name,
                fill="toself",
                opacity=0.6,
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                gridcolor="rgba(255,255,255,0.3)",
                linecolor="rgba(255,255,255,0.3)",
                showgrid=True,
                showline=True,
                ticks="",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.3)",
                linecolor="rgba(255,255,255,0.3)",
                showgrid=True,
                showline=True,
                ticks="",
            ),
        ),
    )

    st.plotly_chart(fig)


def show_spider_plots_per_generation_mmlu_groups(
    experiment_data: ExperimentData,
    selection_type: str,
    generation: int,
    top_N: int,
    benchmark_name: str,
):

    if benchmark_name == "mmlu_pro_llama":
        name = "MMLU Pro"
    else:
        name = "MMLU"

    st.subheader(
        f"{name} Group Scores for Top {top_N} Models in Generation {generation}"
    )
    use_llm_as_a_judge = st.checkbox(
        "Look at LLM-as-a-judge variant",
        value=False,
        key=f"use_llm_as_a_judge_{benchmark_name}",
    )

    # Get the selected models for the selected generation
    top_N_model_names = experiment_data.get_top_models_for_selection_type(
        selection_type, generation, top_N
    )

    # Get the benchmark scores for the selected models
    model_to_mmlu_group_scores = {model_name: {} for model_name in top_N_model_names}

    for model_name in top_N_model_names:
        model_to_mmlu_group_scores[model_name] = (
            experiment_data.get_mmlu_group_scores_for_model(
                model_name=model_name,
                use_llm_as_a_judge=use_llm_as_a_judge,
                benchmark_name=benchmark_name,
            )
        )

    # Create a spider plot for each model where each models spider plot
    # is overlayed on top of the other models spider plots
    fig = go.Figure()
    for model_name, mmlu_group_scores in model_to_mmlu_group_scores.items():
        fig.add_trace(
            go.Scatterpolar(
                r=list(mmlu_group_scores.values()),
                theta=list(mmlu_group_scores.keys()),
                name=model_name,
                fill="toself",
                opacity=0.6,
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                gridcolor="rgba(255,255,255,0.3)",
                linecolor="rgba(255,255,255,0.3)",
                showgrid=True,
                showline=True,
                ticks="",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.3)",
                linecolor="rgba(255,255,255,0.3)",
                showgrid=True,
                showline=True,
                ticks="",
            ),
        ),
    )

    st.plotly_chart(fig)


def show_tSNE_plot_for_task(
    experiment_data: ExperimentData,
    task_name_to_tSNE_embedding: dict,
    selection_type: str = "global_skill_vector_coverage",
    top_N: int = 8,
):
    """Shows the tSNE plot for a specific task."""

    ### Create tSNE plot object ########################################
    # Graph object without edges
    G = get_task_tSNE_graph_object(task_name_to_tSNE_embedding)

    ### Add task selection dropdown ####################################
    # Get all available task names
    all_task_names = list(G.nodes())
    all_task_names.sort()  # Sort for better UX

    # Add "None" option for no selection
    task_options = ["None"] + all_task_names

    # Create dropdown for task selection
    selected_task = st.selectbox(
        "Select a task to highlight:",
        options=task_options,
        index=0,  # Default to "None"
        help="Choose a task to highlight in the tSNE plot",
        key="selected_task_for_tSNE_plot",
    )

    # Convert "None" back to None
    if selected_task == "None":
        selected_task = None

    ### Coloring based on models skill vectors #########################
    # get relevant models based on selection method
    relevant_models = set(
        experiment_data.get_top_models_for_selection_type(
            selection_type, generation=None, top_N=top_N
        )
    )

    # get models to skill vector mapping
    all_models_to_skill_vector = experiment_data.all_models_to_skill_vector

    # get the skill vector for the relevant models
    relevant_models_skill_vectors = {
        model_name: all_models_to_skill_vector[model_name]
        for model_name in relevant_models
    }

    ### Color the nodes based on the models skill vectors ##############
    # Create a color map for the models, where each model has a unique color
    color_map = {
        model_name: COLOR_MAP[i]
        for i, model_name in enumerate(relevant_models, start=1)
    }

    ### Create a color map for the task nodes, where each task has a unique color
    # First, see what colors are asigned to the task nodes in the tasks
    # If a model has solved a task, it is assigned the color of the model
    task_name_to_colors = {}
    task_name_to_solved_by_models = {}
    for model_name, skill_vector in relevant_models_skill_vectors.items():
        for task_name, succesfully_solved in skill_vector.items():
            # Update colors for task nodes
            if task_name not in task_name_to_colors:
                task_name_to_colors[task_name] = []
            if succesfully_solved:
                task_name_to_colors[task_name].append(color_map[model_name])

            # Update models that have solved the task
            if task_name not in task_name_to_solved_by_models:
                task_name_to_solved_by_models[task_name] = []
            if succesfully_solved:
                task_name_to_solved_by_models[task_name].append(model_name)

    # Now, for each task, if there are multiple colors, mix them
    # for task_name, colors in task_name_to_colors.items():
    #     if len(colors) > 1:
    #         task_name_to_colors[task_name] = [mix_colors(colors)]
    # For each task, if there are multiple colors, use the last color in the color list
    for task_name, colors in task_name_to_colors.items():
        if len(colors) > 1:
            task_name_to_colors[task_name] = [COLOR_MAP[-2]]

    # Add default color to tasks that have not been solved by any model
    for task_name in G.nodes():
        if (
            task_name not in task_name_to_colors
            or len(task_name_to_colors[task_name]) == 0
        ):
            task_name_to_colors[task_name] = [COLOR_MAP[-1]]  #  red

    ### Create plotly chart ############################################
    # Create a plotly chart with the tSNE plot object
    # For each node, add a hover text with the task name and the models that have solved it
    fig = go.Figure()

    # Separate data for red nodes, non-red nodes, and highlighted node
    red_x_values = []
    red_y_values = []
    red_hovertext = []
    non_red_x_values = []
    non_red_y_values = []
    non_red_colors = []
    non_red_hovertext = []
    highlighted_x_values = []
    highlighted_y_values = []
    highlighted_hovertext = []

    for task_name in G.nodes():
        x_val = G.nodes[task_name]["embedding"][0]
        y_val = G.nodes[task_name]["embedding"][1]
        color = task_name_to_colors[task_name][0]

        if task_name not in task_name_to_solved_by_models:
            task_name_to_solved_by_models[task_name] = []

        hover_text = build_task_hover_text(
            experiment_path=experiment_data.experiment_path,
            task_name=task_name,
            solved_by_models=task_name_to_solved_by_models[task_name],
        )

        # Check if this is the selected task
        if selected_task and task_name == selected_task:
            highlighted_x_values.append(x_val)
            highlighted_y_values.append(y_val)
            highlighted_hovertext.append(hover_text)
        # Separate red nodes (COLOR_MAP[9]) from other nodes
        elif color == COLOR_MAP[-1]:  # Red nodes
            red_x_values.append(x_val)
            red_y_values.append(y_val)
            red_hovertext.append(hover_text)
        else:  # Non-red nodes
            non_red_x_values.append(x_val)
            non_red_y_values.append(y_val)
            non_red_colors.append(color)
            non_red_hovertext.append(hover_text)

    # Add trace for non-red nodes with opacity 0.5
    if non_red_x_values:
        fig.add_trace(
            go.Scatter(
                x=non_red_x_values,
                y=non_red_y_values,
                mode="markers",
                marker=dict(
                    color=non_red_colors,
                    size=10,
                ),
                hovertext=non_red_hovertext,
                opacity=0.5,
                showlegend=False,
            )
        )

    # Add trace for red nodes with opacity 0.1
    if red_x_values:
        fig.add_trace(
            go.Scatter(
                x=red_x_values,
                y=red_y_values,
                mode="markers",
                marker=dict(
                    color=COLOR_MAP[-1],  # Red
                    size=10,
                ),
                hovertext=red_hovertext,
                opacity=0.10,
                name="Unsolved tasks",
                showlegend=False,
            )
        )

    # Add trace for highlighted task with special styling
    if highlighted_x_values:
        fig.add_trace(
            go.Scatter(
                x=highlighted_x_values,
                y=highlighted_y_values,
                mode="markers",
                marker=dict(
                    color=task_name_to_colors[selected_task][0],
                    size=13,  # Larger size
                    line=dict(color="black", width=1),
                    symbol="diamond",  # Different symbol to make it stand out
                ),
                hovertext=highlighted_hovertext,
                opacity=1.0,
                name=f"Selected: {selected_task}",
                showlegend=False,
            )
        )

    st.header("tSNE Plot")

    # Show the mapping of model names to colors (where the color is rendered as a square)
    st.write("Model to Color Mapping:")
    text_to_write = []
    for model_name, color in color_map.items():
        text_to_write.append(
            f"{model_name}: <div style='width: 10px; height: 10px; background-color: {color}; display: inline-block;'></div><br>"
        )
    # Unsolved tasks are red
    text_to_write.append(
        f"Unsolved tasks: <div style='width: 10px; height: 10px; background-color: {COLOR_MAP[-1]}; display: inline-block;'></div><br>"
    )

    # Tasks where more than 1 model has solved it are grey
    text_to_write.append(
        f"Solved by more than 1 model: <div style='width: 10px; height: 10px; background-color: {COLOR_MAP[-2]}; display: inline-block;'></div><br>"
    )

    # Add legend for highlighted task
    # if selected_task:
    #     text_to_write.append(
    #         f"Selected task: <div style='width: 10px; height: 10px; background-color: {task_name_to_colors[selected_task][0]}; display: inline-block; border: 2px solid black;'></div> (diamond shape, larger size)"
    #     )

    st.write("".join(text_to_write), unsafe_allow_html=True)

    st.plotly_chart(fig)


def show_single_answer_from_pop_data(
    experiment_data: ExperimentData,
    baselines_single_answer_from_pop_data: dict,
    best_of_N_results_per_benchmark_per_selection_method: dict,
    baselines_results_per_benchmark: dict,
    x_axis_variable: str = "selection_method",
    show_ref_best_of_N_results: bool = True,
    task_force_selection_method: str = "global_skill_vector_coverage",
):
    """Show the single answer from pop data.

    Args:
        experiment_data: The experiment data.
        baselines_single_answer_from_pop_data: The baselines single answer from pop data.
        best_of_N_results_per_benchmark_per_selection_method: The best-of-N results per benchmark per selection method.
        baselines_results_per_benchmark: The baselines results per benchmark.
        x_axis_variable: The variable to use on the x-axis.
        show_ref_best_of_N_results: Whether to show the reference best-of-N results.

    Plots a grid of plots for each benchmark.
    Where one big plot is the accuracy scores per model group and per selection method.
    The small plots are the model distributions per model group, where each model has a bar for each selection method.
    """

    def single_row_of_plots(
        benchmark_name: str,
        single_answer_from_pop_data: dict,
        best_of_N_results: dict,
        baselines_results: dict,
        x_axis_variable: str = "selection_method",
        show_ref_best_of_N_results: bool = True,
        task_force_selection_method: str = "global_skill_vector_coverage",
    ):
        """Show a single row of plots for a single benchmark."""

        def get_ref_best_of_N_results(
            best_of_N_results: dict,
            baselines_results: dict,
            task_force_selection_method: str = "global_skill_vector_coverage",
        ):
            """Get the best-of-N results for the reference model group."""
            ref_best_of_N_results = {}
            # N=3 and N=8 for our models
            n3 = best_of_N_results[task_force_selection_method][3]["best_of_N_accuracy"]
            n8 = best_of_N_results[task_force_selection_method][8]["best_of_N_accuracy"]

            # baseline 3 experts and 8x8B-Instruct models
            baseline_3_experts = baselines_results["3_experts"]["best_of_N_accuracy"]
            baseline_8x8B_Instruct = baselines_results["control"][8][
                "best_of_N_accuracy"
            ]
            # big models
            big_model_dict = baselines_results["big_model"]
            all_big_model_results = {}
            for model_size, big_model_results in big_model_dict.items():
                all_big_model_results[f"big_model_{model_size}B"] = big_model_results[
                    "best_of_N_accuracy"
                ]

            ref_best_of_N_results = {
                "N3": n3,
                "N8": n8,
                "experts_N3": baseline_3_experts,
                "control_N8": baseline_8x8B_Instruct,
                **all_big_model_results,
            }
            return ref_best_of_N_results

        def get_model_group_color_from_fig(fig: go.Figure, model_group_name: str):
            """Get the color used for the model group from the fig object."""
            for trace in fig.data:
                if trace.name == model_group_name:
                    return trace.marker.color
            return None

        st.subheader(f"{multi_word_string_to_readable_string(benchmark_name)}")

        # Get the model groups for the benchmark
        model_groups_ours: dict[str, dict] = single_answer_from_pop_data.get(
            benchmark_name, {}
        )

        model_groups_baselines: dict[str, dict] = (
            baselines_single_answer_from_pop_data.get(benchmark_name, {})
        )

        model_groups = {
            **model_groups_ours,
            **model_groups_baselines,
        }

        # Get the model names for the benchmark
        model_names = set()
        for model_group in model_groups.values():
            for model_name in model_group["model_distribution"].keys():
                model_names.add(model_name)

        ### Accuracy scores and plot ###################################
        # Get the accuracy scores for the benchmark, per model group and selection method
        accuracy_scores = {}
        flattened_accuracy_scores = []
        flattened_model_group_names = []
        flattened_selection_methods = []
        for model_group_name, model_group in model_groups.items():
            if model_group_name not in accuracy_scores:
                accuracy_scores[model_group_name] = {}
            for selection_method, score in model_group["results"].items():
                accuracy_scores[model_group_name][selection_method] = score
                flattened_accuracy_scores.append(score)
                flattened_model_group_names.append(model_group_name)
                flattened_selection_methods.append(selection_method)

        # Selection_method to unique index
        selection_method_to_idx = {}
        for idx, selection_method in enumerate(set(flattened_selection_methods)):
            selection_method_to_idx[selection_method] = f"Method {idx+1}"

        data_df = pd.DataFrame(
            {
                "model_group": flattened_model_group_names,
                "selection_method": flattened_selection_methods,
                "accuracy_score": flattened_accuracy_scores,
            }
        )

        if x_axis_variable == "selection_method":
            color_variable = "model_group"
        elif x_axis_variable == "model_group":
            color_variable = "selection_method"

        fig = px.bar(
            data_df,
            x=x_axis_variable,
            y="accuracy_score",
            color=color_variable,
            title="Accuracy Scores per Model Group and Selection Method",
            barmode="group",
            labels={
                "model_group": "Model Group",
                "selection_method": "Selection Method",
                "accuracy_score": "Accuracy Score",
            },
        )

        # start the y-axis at the lowest accuracy score - 10%
        min_y_axis = min(data_df["accuracy_score"]) - 0.1
        # end the y-axis at the highest accuracy score + 10%
        max_y_axis = max(data_df["accuracy_score"]) + 0.1

        if show_ref_best_of_N_results:
            # Add horizontal lines for reference best-of-N results to the plot
            ref_best_of_N_results = get_ref_best_of_N_results(
                best_of_N_results=best_of_N_results,
                baselines_results=baselines_results,
                task_force_selection_method=task_force_selection_method,
            )
            for model_group_name, score in ref_best_of_N_results.items():
                # get the color used for the model group from the fig object
                if x_axis_variable == "selection_method":
                    color = get_model_group_color_from_fig(fig, model_group_name)
                else:
                    # random color
                    random_color_idx = random.randint(0, len(COLOR_MAP) - 1)
                    color = COLOR_MAP[random_color_idx]
                fig.add_hline(
                    y=score,
                    line_width=2,
                    line_dash="dash",
                    opacity=0.5,
                    annotation_text=model_group_name,
                    annotation_position="top right",
                    line_color=color,
                )

                if score < min_y_axis:
                    min_y_axis = score - 0.05
                if score > max_y_axis:
                    max_y_axis = score + 0.05

        fig.update_yaxes(range=[min_y_axis, max_y_axis])

        st.plotly_chart(fig)

        ### Model Distribution #########################################
        # Get the model distribution for the benchmark, per model group and selection method
        data_df_per_model_group = {}

        for model_group_name, model_group in model_groups.items():
            for model_name, distribution_over_models in model_group[
                "model_distribution"
            ].items():
                if model_group_name not in data_df_per_model_group:
                    data_df_per_model_group[model_group_name] = []
                for (
                    selection_method,
                    selected_model_percentage,
                ) in distribution_over_models.items():
                    data_df_per_model_group[model_group_name].append(
                        {
                            "model_name": model_name,
                            "selection_method": selection_method,
                            # "selection_method": selection_method_to_idx[
                            #     selection_method
                            # ],
                            "model_distribution": selected_model_percentage,
                        }
                    )
            # convert to dataframe
            data_df_per_model_group[model_group_name] = pd.DataFrame(
                data_df_per_model_group[model_group_name]
            )

        # Create a 2x2 grid of plots for each model group
        n_model_groups = len(data_df_per_model_group)
        cols = 2
        rows = n_model_groups // cols + (n_model_groups % cols > 0)

        # Color map from model name to color
        model_name_to_color = {}
        for i, model_name in enumerate(model_names):
            model_name_to_color[model_name] = COLOR_MAP[i % len(COLOR_MAP)]

        # Create individual plots for each model group
        for i, (model_group_name, data_df) in enumerate(
            data_df_per_model_group.items()
        ):
            fig = px.bar(
                data_df,
                x="selection_method",
                y="model_distribution",
                color="model_name",
                title=f"Model Distribution for {multi_word_string_to_readable_string(model_group_name)}",
                labels={
                    "model_distribution": "Usage Count",
                    "selection_method": "Selection Method",
                    "model_name": "Model",
                },
                color_discrete_map=model_name_to_color,
            )

            # Update layout to show legend for each plot
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                ),
                margin=dict(r=150),  # Add margin for legend
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

    # Create a grid of plots for each benchmark
    single_answer_from_pop_data = experiment_data.single_answer_from_pop_data[
        task_force_selection_method
    ]
    for benchmark_name in single_answer_from_pop_data.keys():
        # Get the best-of-N results for the benchmark
        best_of_N_results = best_of_N_results_per_benchmark_per_selection_method[
            benchmark_name
        ]
        baselines_results = baselines_results_per_benchmark[benchmark_name]
        single_row_of_plots(
            benchmark_name=benchmark_name,
            single_answer_from_pop_data=single_answer_from_pop_data,
            best_of_N_results=best_of_N_results,
            baselines_results=baselines_results,
            x_axis_variable=x_axis_variable,
            show_ref_best_of_N_results=show_ref_best_of_N_results,
            task_force_selection_method=task_force_selection_method,
        )


def show_continuous_coevolution_scaling_law_plot(
    experiment_data: ExperimentData,
    relevant_benchmarks: list[str] = None,
    do_log_scale: bool = False,
    selected_N: list[int] = None,
    polynomial_degree: int = 1,
    relevant_file_pattern: str = "gsvc_max_gen_*_w_task_filtering",
):
    """Show the continuous coevolution scaling law plots."""

    relevant_selection_methods = glob.glob(
        os.path.join(
            experiment_data.experiment_path,
            "eval",
            "best_of_N",
            experiment_data.existing_benchmarks[0],
            relevant_file_pattern,
        )
    )
    relevant_selection_methods = [
        os.path.basename(selection_method)
        for selection_method in relevant_selection_methods
    ]
    # Sort by gen number
    relevant_selection_methods = sorted(
        relevant_selection_methods,
        key=lambda x: int(x.split("_")[3]),
    )

    (
        best_of_N_results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results,
    ) = get_best_of_N_results_per_benchmark_per_selection_method(
        best_of_N_results_dir_path=os.path.join(
            experiment_data.experiment_path, "eval", "best_of_N"
        ),
        relevant_selection_methods=relevant_selection_methods,
        relevant_benchmarks=relevant_benchmarks,
    )

    continuous_coevolution_scaling_law_data = get_continuous_coevolution_scaling_law_data(
        best_of_N_results_per_benchmark_per_selection_method=best_of_N_results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results=averaged_across_benchmarks_results,
    )

    # Get plots for the continuous coevolution scaling law data
    # for each benchmark and averaged across benchmarks
    benchmark_to_figure = {}
    for benchmark_name, data in continuous_coevolution_scaling_law_data.items():
        benchmark_to_figure[benchmark_name] = (
            get_continuous_coevolution_scaling_law_plot(
                continuous_coevolution_scaling_law_data=data,
                benchmark_name=benchmark_name,
                do_log_scale=do_log_scale,
                selected_N=selected_N,
                polynomial_degree=polynomial_degree,
            )
        )

    # First, plot the averaged across benchmarks
    st.plotly_chart(
        benchmark_to_figure["averaged_across_benchmarks"],
        use_container_width=True,
    )

    # Just plot all other benchmarks
    for benchmark_name, figure in benchmark_to_figure.items():
        if benchmark_name == "averaged_across_benchmarks":
            continue
        st.plotly_chart(figure, use_container_width=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the evolution of model answers over generations."
    )
    parser.add_argument(
        "experiment_path_and_baselines_results_dir_path",
        type=str,
        help="Path to the experiment directory containing evaluation results for multiple generations and baselines results directory.",
    )
    return parser.parse_args()


def main(experiment_path, baselines_results_dir_path):
    print(f"Experiment path: {experiment_path}")
    print(f"Baselines results dir path: {baselines_results_dir_path}")
    st.title("Model Archive Analysis")
    # Add experiment info
    experiment_name = "/".join(experiment_path.split("/")[-2:])
    st.write(f"**Experiment:** {experiment_name}")

    ### Initialize experiment data with caching ########################
    (
        experiment_data,
        task_name_to_tSNE_embedding,
        task_name_to_hdbscan_cluster,
    ) = load_experiment_data(experiment_path)
    relevant_benchmarks = experiment_data.existing_benchmarks

    ### Load baselines results #########################################
    (
        baselines_results_experts,
        baselines_results_big_model,
        baselines_results_control,
        baselines_single_answer_from_pop_data,
    ) = load_baselines_results(
        baselines_results_dir_path=baselines_results_dir_path,
        relevant_benchmarks=relevant_benchmarks,
    )

    ### Create Generation Tree #########################################
    if experiment_data.models_to_parent_models:
        try:
            G = create_generation_tree(
                models_to_parent_models=experiment_data.models_to_parent_models,
            )
        except Exception as e:
            st.error(f"Error creating generation tree: {e}")
            G = None
    else:
        G = None
        st.warning(
            "No models_to_parent_models found in experiment data. No generation tree will be created."
        )

    ####################################################################
    ### Best-of-N Plots ################################################
    ####################################################################
    st.header("Best-of-N Accuracy Plots")

    ### Load best-of-N results
    best_of_N_results_dir_path = os.path.join(experiment_path, "eval", "best_of_N")
    # Get all selection methods from one benchmark
    one_benchmark = glob.glob(os.path.join(best_of_N_results_dir_path, "*"))[0]
    selection_methods_in_one_benchmark = glob.glob(os.path.join(one_benchmark, "*"))
    relevant_selection_methods = [
        os.path.basename(selection_method)
        for selection_method in selection_methods_in_one_benchmark
    ]
    # remove "coverage" and "fitness"
    relevant_selection_methods = [
        selection_method
        for selection_method in relevant_selection_methods
        if selection_method != "coverage" and selection_method != "fitness"
    ]

    # Dropdown to select (multiple) selection methods (shortened names)
    selection_method_to_short_name = {
        selection_method: multi_word_string_to_readable_string(selection_method)
        for selection_method in relevant_selection_methods
    }
    selected_selection_methods = st.multiselect(
        "Select Selection Methods",
        options=relevant_selection_methods,
        default=(
            "global_skill_vector_coverage"
            if "global_skill_vector_coverage" in relevant_selection_methods
            else relevant_selection_methods[0]
        ),
        format_func=lambda x: selection_method_to_short_name[x],
    )
    # Print the names of the  selected model per selection method
    # Create a table to display selection methods and their models
    table_data = []
    for selection_method in selected_selection_methods:
        # Get the best_of_N file for the selection method
        best_of_N_file = os.path.join(
            experiment_path,
            "eval",
            "best_of_N",
            relevant_benchmarks[0],
            selection_method,
            "results_N8.json",
        )

        with open(best_of_N_file, "r") as f:
            best_of_N_results = json.load(f)
        model_names = []
        for model_name in best_of_N_results["analyzed_files"]:
            if "__models__" in model_name:
                model_names.append(model_name.split("__models__")[-1])
            else:
                model_names.append(model_name)

        # Sort model names based on gen(gen_X_ind_Y)
        # model_names = sorted(
        #     model_names,
        #     key=lambda x: (int(x.split("_")[1]),),
        # )

        # Add selection method and its models to table data
        row = [selection_method_to_short_name[selection_method]] + model_names
        table_data.append(row)

    # Create and display the table
    if table_data:
        # Get the maximum number of models across all selection methods
        max_models = max(len(row) for row in table_data)
        # Create column names
        columns = ["Selection Method (N=8)"] + [
            f"Model {i+1}" for i in range(max_models - 1)
        ]
        # Pad rows with empty strings if needed
        padded_table_data = [row + [""] * (max_models - len(row)) for row in table_data]
        # Display the table
        st.table(pd.DataFrame(padded_table_data, columns=columns))

    (
        best_of_N_results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results,
        baselines_results_per_benchmark,
        baselines_averaged_across_benchmarks_results,
    ) = get_best_of_N_results_for_main_best_of_N_plots(
        experiment_path=experiment_path,
        baselines_results_dir_path=baselines_results_dir_path,
        relevant_selection_methods=selected_selection_methods,
    )

    show_best_of_N_plots(
        best_of_N_results_per_benchmark_per_selection_method=best_of_N_results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results=averaged_across_benchmarks_results,
        baselines_results_per_benchmark=baselines_results_per_benchmark,
        baselines_averaged_across_benchmarks_results=baselines_averaged_across_benchmarks_results,
    )

    st.subheader("Averaged Across Selected Benchmarks")

    # Dropdown to select (multiple) benchmarks to compute the average metric for
    # Add preset options
    preset_options = {
        "[Preset] All": relevant_benchmarks,
        "[Preset] All OE": [
            "gsm8k_llama",
            "minerva_math",
            "ifeval",
            "mmlu_cot_llama_llm_as_a_judge",
            "mmlu_pro_llama_llm_as_a_judge",
            "bbh_cot_zeroshot_llm_as_a_judge",
            "gpqa_main_cot_zeroshot_llm_as_a_judge",
            "humaneval_instruct",
            "mbpp_instruct",
        ],
        "[Preset] Default OE": ["gsm8k_llama", "minerva_math", "ifeval"],
        "[Preset] All llm-as-a-judge": [
            "mmlu_cot_llama_llm_as_a_judge",
            "mmlu_pro_llama_llm_as_a_judge",
            "bbh_cot_zeroshot_llm_as_a_judge",
            "gpqa_main_cot_zeroshot_llm_as_a_judge",
        ],
        "[Preset] Default MCQ": [
            "gpqa_main_cot_zeroshot",
            "gpqa_diamond_cot_zeroshot",
            "bbh_cot_zeroshot",
            "mmlu_cot_llama",
            "mmlu_pro_llama",
            "arc_challenge_llama",
        ],
        "[Preset] Default math": [
            "gsm8k_llama",
            "minerva_math",
            "gpqa_main_cot_zeroshot",
        ],
    }
    selected_benchmarks = st.multiselect(
        "Select benchmarks for average metric computation",
        options=relevant_benchmarks + list(preset_options.keys()),
        default="[Preset] All",
    )

    if len(selected_benchmarks) > 0:
        error = False
        # When selecting a preset option, you can't select anything else
        if len(selected_benchmarks) > 1 and any(
            option in selected_benchmarks for option in preset_options.keys()
        ):
            st.warning(
                "When selecting a preset option, you can't select any other benchmarks/ preset options"
            )
            error = True
        # When selecting a single preset option, select the preset option
        elif (
            len(selected_benchmarks) == 1
            and selected_benchmarks[0] in preset_options.keys()
        ):
            selected_benchmarks = preset_options[selected_benchmarks[0]]

            # check if selected benchmarks are available
            # remove the benchmarks that we don't have data for
            for benchmark in selected_benchmarks:
                if benchmark not in relevant_benchmarks:
                    st.warning(f"Benchmark {benchmark} is not available")
                    selected_benchmarks.remove(benchmark)

            if len(selected_benchmarks) == 0:
                st.warning("There is no benchmark data for the selected preset option")
                error = True

        if not error:
            show_and_compute_average_metric_for_selected_benchmarks(
                best_of_N_results_dir_path=os.path.join(
                    experiment_path, "eval", "best_of_N"
                ),
                baselines_results_dir_path=baselines_results_dir_path,
                relevant_selection_methods=selected_selection_methods,
                relevant_benchmarks=selected_benchmarks,
            )

    st.divider()

    ### Scaling law plots ##############################################
    st.header("Continuous Coevolution Scaling Law Plots")
    selected_benchmarks = st.multiselect(
        "Select benchmarks for scaling law plots",
        options=relevant_benchmarks + list(preset_options.keys()),
        default="[Preset] All OE",
        key="selected_benchmarks_for_scaling_law_plots",
    )

    try:
        if len(selected_benchmarks) > 0:
            error = False
            # When selecting a preset option, you can't select anything else
            if len(selected_benchmarks) > 1 and any(
                option in selected_benchmarks for option in preset_options.keys()
            ):
                st.warning(
                    "When selecting a preset option, you can't select any other benchmarks/ preset options"
                )
                error = True
            # When selecting a single preset option, select the preset option
            elif (
                len(selected_benchmarks) == 1
                and selected_benchmarks[0] in preset_options.keys()
            ):
                selected_benchmarks = preset_options[selected_benchmarks[0]]

                # check if selected benchmarks are available
                # remove the benchmarks that we don't have data for
                for benchmark in selected_benchmarks:
                    if benchmark not in relevant_benchmarks:
                        st.warning(f"Benchmark {benchmark} is not available")
                        selected_benchmarks.remove(benchmark)

                if len(selected_benchmarks) == 0:
                    st.warning(
                        "There is no benchmark data for the selected preset option"
                    )
                    error = True

            if not error:
                do_log_scale = st.checkbox("Log Scale X-axis", value=False)
                # multiple check boxes to pick which N to use for the scaling law plots
                selected_N = st.multiselect(
                    "Select N for scaling law plots",
                    options=list(range(1, experiment_data.get_max_top_N() + 1)),
                    default=[8],
                    key="selected_N_for_scaling_law_plots",
                )
                polynomial_degree = st.selectbox(
                    "Select Polynomial Degree for regression line",
                    options=[1, 2, 3],
                    key="polynomial_degree_for_regression_line",
                )

                # Dropdown to select relevant file pattern
                relevant_file_pattern = st.selectbox(
                    "Select Relevant File Pattern",
                    options=[
                        "gsvc_max_gen_*_w_task_filtering",
                        "gsvc_max_gen_*_w_task_filtering_w_instruct_llama",  # for backwards compatibility
                        "gsvc_max_gen_*_w_task_filtering_w_instruct",
                        "gsvc_max_gen_*_w_task_filtering_w_instruct_only",
                        "gsvc_max_gen_*_w_task_filtering_w_all_seed",
                    ],
                    key="relevant_file_pattern_for_scaling_law_plots",
                )

                show_continuous_coevolution_scaling_law_plot(
                    experiment_data=experiment_data,
                    relevant_benchmarks=selected_benchmarks,
                    do_log_scale=do_log_scale,
                    selected_N=selected_N,
                    polynomial_degree=polynomial_degree,
                    relevant_file_pattern=relevant_file_pattern,
                )
    except Exception as e:
        st.error(
            f"Error creating scaling law plots: {e}. Maybe you need to run best-of-N script again to get the global skill vector based results per generation?"
        )

    ####################################################################
    ### Single Answer From Pop #########################################
    ####################################################################

    st.header("Single Answer From Pop")

    if (
        baselines_single_answer_from_pop_data
        and experiment_data.single_answer_from_pop_data
    ):
        # try:

        # Dropdown to select task force selection method
        task_force_selection_method = st.selectbox(
            "Select Task Force Selection Method",
            options=list(experiment_data.single_answer_from_pop_data.keys()),
            key="task_force_selection_method_for_single_answer_from_pop",
        )

        # Dropdown to select x axis variable
        x_axis_variable = st.selectbox(
            "Select X Axis Variable for Accuracy Plot",
            options=["selection_method", "model_group"],
            index=0,
            format_func=lambda x: multi_word_string_to_readable_string(x),
        )

        show_ref_best_of_N_results = st.checkbox(
            "Show Reference Best-of-N Results",
            value=True,
        )

        # Show the single answer from pop data
        show_single_answer_from_pop_data(
            experiment_data=experiment_data,
            baselines_single_answer_from_pop_data=baselines_single_answer_from_pop_data,
            best_of_N_results_per_benchmark_per_selection_method=best_of_N_results_per_benchmark_per_selection_method,
            baselines_results_per_benchmark=baselines_results_per_benchmark,
            x_axis_variable=x_axis_variable,
            show_ref_best_of_N_results=show_ref_best_of_N_results,
            task_force_selection_method=task_force_selection_method,
        )
        # except Exception as e:
        #     st.error(f"Error showing single answer from pop data: {e}")
    else:
        st.warning("No single answer from pop data found")

    st.divider()

    ####################################################################
    ### Model Evolution Tree ############################################
    ####################################################################

    st.header("Model Evolution Tree")

    if G:
        # Dropdown to select (multiple) selection methods (shortened names)
        selection_method_to_short_name = {
            selection_method: multi_word_string_to_readable_string(selection_method)
            for selection_method in relevant_selection_methods
        }
        selection_method = st.selectbox(
            "Highlight Models from Selection Methods",
            options=["None"] + relevant_selection_methods,
            format_func=lambda x: (
                selection_method_to_short_name[x] if x != "None" else "None"
            ),
            key="selected_selection_methods_for_evolution_tree",
        )

        # DEBUG
        # selection_method = "global_skill_vector_coverage"

        # Print the names of the  selected model per selection method
        # Create a table to display selection methods and their models
        table_data = []
        top_N_model_names = {}
        if selection_method != "None":
            # Get the best_of_N file for the selection method
            best_of_N_file = os.path.join(
                experiment_path,
                "eval",
                "best_of_N",
                relevant_benchmarks[0],
                selection_method,
                "results_N8.json",
            )

            with open(best_of_N_file, "r") as f:
                best_of_N_results = json.load(f)
            for model_name in best_of_N_results["analyzed_files"]:
                model_name = model_name.split("__models__")[-1]
                top_N_model_names[model_name] = compute_fitness_from_skill_vector(
                    list(
                        experiment_data.all_models_to_skill_vector[model_name].values()
                    )
                )

            # Sort the dictionary by gen number
            top_N_model_names = dict(
                sorted(
                    top_N_model_names.items(),
                    key=lambda x: (int(x[0].split("_")[1]),),
                )
            )

            # Add selection method and its models to table data
            row = [selection_method_to_short_name[selection_method]] + list(
                top_N_model_names.keys()
            )
            table_data.append(row)

        # Create and display the table
        if table_data:
            # Get the maximum number of models across all selection methods
            max_models = max(len(row) for row in table_data)
            # Create column names
            columns = ["Selection Method (N=8)"] + [
                f"Model {i+1}" for i in range(max_models - 1)
            ]
            # Pad rows with empty strings if needed
            padded_table_data = [
                row + [""] * (max_models - len(row)) for row in table_data
            ]
            # Display the table
            st.table(pd.DataFrame(padded_table_data, columns=columns))

        # Get all available models for selection
        all_models = list(experiment_data.models_to_parent_models.keys())

        if all_models:
            # Create a dropdown to select a model to highlight
            selected_model_for_tree = st.selectbox(
                "Select a model to highlight in the evolution tree (optional)",
                options=["None"] + sorted(all_models),
                format_func=lambda x: (
                    x if x == "None" else f"{x} (Gen {x.split('_')[1]})"
                ),
            )

            if selected_model_for_tree != "None":
                selected_model_for_tree = selected_model_for_tree.split(" (Gen")[0]
            else:
                selected_model_for_tree = None

            # DEBUG: select a model for the tree
            # selected_model_for_tree = "gen_38_ind_5"

            # Create and display the evolution tree
            try:
                if selected_model_for_tree and top_N_model_names:
                    top_N_model_names = {}
                    st.warning(
                        "Prioritizing highlighting a specific model over top N models"
                    )

                # Calculate positions once and store them in the session state
                if "tree_positions" not in st.session_state:
                    (st.session_state.tree_positions, _) = (
                        create_equally_spaced_positions(G)
                    )

                fig = create_interactive_generation_tree(
                    G=G,
                    models_to_parent_models=experiment_data.models_to_parent_models,
                    selected_model=selected_model_for_tree,
                    top_N_model_names=top_N_model_names,
                    positions=st.session_state.tree_positions,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating evolution tree: {e}")
        else:
            st.warning("No evolved models found for tree visualization")

        st.divider()
    else:
        st.warning("No model evolution tree data found.")

    ####################################################################
    ### tSNE Plot ######################################################
    ####################################################################

    # TODO: Add slider for top N and drop down for selection type
    if task_name_to_tSNE_embedding:
        # Dropdown to select (multiple) selection methods (shortened names)
        selection_method_to_short_name = {
            selection_method: multi_word_string_to_readable_string(selection_method)
            for selection_method in relevant_selection_methods
        }
        selection_method = st.selectbox(
            "Select Selection Method for Task Force highligted in tSNE Plot",
            options=relevant_selection_methods,
            format_func=lambda x: (selection_method_to_short_name[x]),
            key="selected_selection_methods_for_tSNE_plot",
        )

        show_tSNE_plot_for_task(
            experiment_data=experiment_data,
            task_name_to_tSNE_embedding=task_name_to_tSNE_embedding,
            selection_type=selection_method,
            top_N=8,
        )

        st.divider()

    ####################################################################
    ### Metric Evolution ###############################################
    ####################################################################

    st.header(f"Evolution of Metrics over Generations")

    # Check box to see if wse want to see the evolution of metrics for the selection methods
    # that are available in the experiment
    show_metric_evolution_for_selection_methods = st.checkbox(
        "Show metric evolution for selection methods",
        value=False,
        key="show_metric_evolution_for_selection_methods_checkbox",
    )

    if show_metric_evolution_for_selection_methods:
        # Get all selection methods from one benchmark
        one_benchmark = glob.glob(os.path.join(best_of_N_results_dir_path, "*"))[0]
        selection_methods_in_one_benchmark = glob.glob(os.path.join(one_benchmark, "*"))
        relevant_selection_methods = [
            os.path.basename(selection_method)
            for selection_method in selection_methods_in_one_benchmark
        ]

        if "coverage" and "fitness" in relevant_selection_methods:

            # Add selection criteria buttons
            select_based_on_metric_evolution = st.radio(
                "Select based on:",
                options=["coverage", "fitness"],
                horizontal=True,
                key="select_based_on_metric_evolution",
            )

            # Select top N
            top_N_metric_evolution = st.select_slider(
                "Select Top N",
                options=list(range(1, experiment_data.get_max_top_N() + 1)),
                value=1,
                key="top_N_metric_evolution",
            )

            # Show metric evolution per generation
            show_metric_evolution_per_generation(
                experiment_data=experiment_data,
                selection_type=select_based_on_metric_evolution,
                baselines_results_experts=baselines_results_experts,
                baselines_results_big_model=baselines_results_big_model,
                baselines_results_control=baselines_results_control,
                top_N=top_N_metric_evolution,
            )

        else:
            st.warning("No coverage or fitness per generation selection methods found")

    st.divider()

    ####################################################################
    ### Spider Plots ###################################################
    ####################################################################

    st.header(f"Spider Plots")

    # Dropdown for single selection strategy
    all_selection_methods = relevant_selection_methods + ["coverage", "fitness"]
    selection_strategy = st.selectbox(
        "Select Selection Strategy",
        options=all_selection_methods,
        format_func=lambda x: multi_word_string_to_readable_string(x),
    )

    # Select top N
    top_N_spider_plots = st.select_slider(
        "Select Top N",
        options=list(range(1, experiment_data.get_max_top_N() + 1)),
        value=1,
        key="top_N_spider_plots",
    )

    # Only show generation slider for coverage and fitness
    if selection_strategy in ["coverage", "fitness"]:
        selected_gen_spider_plots = st.select_slider(
            "Select Generation",
            options=experiment_data.get_all_available_gens(),
            value=min(experiment_data.get_all_available_gens()),
        )
    else:
        # For other selection strategies, use the latest generation
        selected_gen_spider_plots = None

    show_spider_plots_per_model(
        experiment_data=experiment_data,
        selection_type=selection_strategy,
        generation=selected_gen_spider_plots,
        top_N=top_N_spider_plots,
    )

    if "mmlu_cot_llama" in experiment_data.existing_benchmarks:
        show_spider_plots_per_generation_mmlu_groups(
            experiment_data=experiment_data,
            selection_type=selection_strategy,
            generation=selected_gen_spider_plots,
            top_N=top_N_spider_plots,
            benchmark_name="mmlu_cot_llama",
        )

    if "mmlu_pro_llama" in experiment_data.existing_benchmarks:
        show_spider_plots_per_generation_mmlu_groups(
            experiment_data=experiment_data,
            selection_type=selection_strategy,
            generation=selected_gen_spider_plots,
            top_N=top_N_spider_plots,
            benchmark_name="mmlu_pro_llama",
        )
    st.divider()

    ####################################################################
    ### Archive Output Comparison ######################################
    ####################################################################

    st.header("Detailed Model Output Analysis")

    # Check box to see if we want to see the archive output comparison
    show_archive_output_comparison = st.checkbox(
        "Show detailed model output analysis",
        value=False,
        key="show_detailed_model_output_analysis_checkbox",
    )

    if show_archive_output_comparison:
        st.error(
            "This feature is currently deactivated, since we are not loading the model eval deatails for speedups."
        )
        return

        # Select top N
        top_N_archive_output_comparison = st.select_slider(
            "Select Top N",
            options=list(range(1, experiment_data.get_max_top_N() + 1)),
            value=1,
            key="top_N_archive_output_comparison",
        )

        # Select benchmark
        selected_benchmark = st.selectbox(
            "Select Benchmark", options=experiment_data.existing_benchmarks
        )

        # Select task
        selected_task = st.selectbox(
            "Select Specific Task To Inspect",
            options=experiment_data.get_task_ids(selected_benchmark),
        )

        st.subheader("Archive Output Comparison")

        ### Answers for selected models given selection method
        st.subheader("Answers for selected models given selection method")
        # pick selection method
        selected_selection_method = st.selectbox(
            "Select Selection Method",
            options=relevant_selection_methods,
            format_func=lambda x: multi_word_string_to_readable_string(x),
        )

        # Get models for selected selection method
        model_examples = (
            experiment_data.get_model_examples_for_selection_type_task_and_generation(
                task_id=selected_task,
                generation=None,
                selection_type=selected_selection_method,
                top_N=top_N_archive_output_comparison,
                benchmark=selected_benchmark,
            )
        )
        # Show model answers for selected task
        show_model_answers(
            model_1_examples=model_examples,
            task_id=selected_task,
        )

        ### Comparison across generations
        st.subheader(
            f"Inspecting Task '{selected_task}' in Benchmark '{selected_benchmark}'"
        )
        # Add selection criteria buttons
        select_based_on_archive_output_comparison = st.radio(
            "Select based on:",
            options=["coverage", "fitness"],
            horizontal=True,
            key="select_based_on_archive_output_comparison",
        )

        try:
            # Show evolution chart for the selected task
            correct_solutions = (
                experiment_data.get_number_of_correct_solutions_per_generation(
                    selection_type=select_based_on_archive_output_comparison,
                    benchmark=selected_benchmark,
                    task_id=selected_task,
                    top_N=top_N_archive_output_comparison,
                )
            )
            show_evolution_chart(
                correct_solutions=correct_solutions,
                top_N=top_N_archive_output_comparison,
            )
        except Exception as e:
            st.error(
                f"Error showing evolution chart: {e}\nMaybe no best-of-N results for `coverage` and `fitness`?"
            )

        st.subheader("Comparison across generations")

        try:
            # Select generations
            gen_nums = experiment_data.get_all_available_gens()
            col1, col2 = st.columns(2)
            with col1:
                selected_gen1 = st.select_slider(
                    "Select First Generation",
                    options=gen_nums,
                    value=min(gen_nums),
                    key="selected_gen1",
                )
            with col2:
                selected_gen2 = st.select_slider(
                    "Select Second Generation",
                    options=gen_nums,
                    value=max(gen_nums),
                    key="selected_gen2",
                )

            # Get model examples data for both generations
            model_1_examples = experiment_data.get_model_examples_for_selection_type_task_and_generation(
                task_id=selected_task,
                generation=selected_gen1,
                selection_type=select_based_on_archive_output_comparison,
                top_N=top_N_archive_output_comparison,
                benchmark=selected_benchmark,
            )
            model_2_examples = experiment_data.get_model_examples_for_selection_type_task_and_generation(
                task_id=selected_task,
                generation=selected_gen2,
                selection_type=select_based_on_archive_output_comparison,
                top_N=top_N_archive_output_comparison,
                benchmark=selected_benchmark,
            )

            if not model_1_examples or not model_2_examples:
                st.error(f"No model examples found for {selected_benchmark}")
                return

            # Show model answers for selected task
            show_model_answers(
                model_1_examples=model_1_examples,
                model_2_examples=model_2_examples,
                generation1=selected_gen1,
                generation2=selected_gen2,
                task_id=selected_task,
            )
        except Exception as e:
            st.error(
                f"Error showing model answers per gen: {e}\nMaybe no best-of-N results for `coverage` and `fitness`?"
            )


if __name__ == "__main__":
    args = parse_args()
    experiment_path, baselines_results_dir_path = (
        args.experiment_path_and_baselines_results_dir_path.split(",")
    )
    main(experiment_path, baselines_results_dir_path)
