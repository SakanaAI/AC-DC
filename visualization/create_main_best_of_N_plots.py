import os
import json
import logging
import glob
import matplotlib.pyplot as plt
import re

from utils import COLOR_PALETTE

logger = logging.getLogger(__name__)


def get_baselines_results(
    baselines_results_dir_path: str, relevant_benchmarks: list[str]
) -> dict:
    """
    Get the baseline results from the baselines directory.
    The baselines directory has the following structure:
    <baselines_results_dir_path>/
    ├── <benchmark_name>/
    │   ├── 70B_results_N1.json
    │   ├── results_N3.json # 3 experts
    │   ├── control_results_N1.json
    │   ├── control_results_N2.json
    │   ├── ...
    ├── <benchmark_name>/
    │   ├── 70B_results_N1.json
    │   ├── results_N3.json # 3 experts
    │   ├── control_results_N1.json
    │   ├── control_results_N2.json
    │   ├── ...
    ├── ...

    Returns:
        dict[str, dict[str, dict[str, dict[str, float]]]]: A dictionary of dictionaries containing the baseline results per benchmark per selection method.
        In the format of:
        {
            <benchmark_name>: {
                llama_70B: {
                    "best_of_N_accuracy": float,
                    "majority_vote_accuracy": float,
                    "unique_contributions": {
                        <model_name>: int,
                        ...
                    },
                    "num_unique_samples": int,
                },
                3_experts: {
                    "best_of_N_accuracy": float,
                    "majority_vote_accuracy": float,
                    "unique_contributions": {
                        <model_name>: int,
                        ...
                    },
                    "num_unique_samples": int,
                },
                control: {
                    <N>: {
                        "best_of_N_accuracy": float,
                        "majority_vote_accuracy": float,
                        "unique_contributions": {
                            <model_name>: int,
                            ...
                        },
                        "num_unique_samples": int,
                    },
                    ...
                },
            },
            ...
        }
        Average over the benchmarks.
        {
            <benchmark_name>: {
                "llama_70B": {
                    "best_of_N_accuracy": float,
                    "majority_vote_accuracy": float,
                },
                "3_experts": {
                    "best_of_N_accuracy": float,
                    "majority_vote_accuracy": float,
                },
                "control": {
                    <N>: {
                        "best_of_N_accuracy": float,
                        "majority_vote_accuracy": float,
                    },
                    ...
                },
                ...
            },
            ...
        }
    """
    results_per_benchmark = {}
    averaged_across_benchmarks_results = {}

    # Iterate through each benchmark directory
    for benchmark_dir in relevant_benchmarks:
        benchmark_path = os.path.join(baselines_results_dir_path, benchmark_dir)
        if not os.path.isdir(benchmark_path):
            logger.warning(f"Benchmark directory {benchmark_path} not found.")
            continue

        results_per_benchmark[benchmark_dir] = {
            "big_model": {},
            "3_experts": {},
            "control": {},
        }

        # Process all JSON files in the benchmark directory
        for file_name in os.listdir(benchmark_path):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(benchmark_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract N from filename (e.g., results_N3.json -> 3)
            N = int(file_name.split("_N")[-1].split(".")[0])

            # Determine which category this result belongs to
            # Check for 70B for backwards compatibility
            if "big_model" in file_name or "70B" in file_name:
                model_size_regex = r"(\d+)[Bb]"
                model_size_match = re.search(model_size_regex, file_name)
                if model_size_match:
                    model_size = model_size_match.group(1)
                else:
                    logger.warning(f"No model size found in {file_name}")
                    continue
                if model_size not in results_per_benchmark[benchmark_dir]["big_model"]:
                    results_per_benchmark[benchmark_dir]["big_model"][model_size] = {}
                results_per_benchmark[benchmark_dir]["big_model"][model_size] = {
                    "best_of_N_accuracy": data.get("best_of_N_accuracy", 0.0),
                    "majority_vote_accuracy": data.get("majority_vote_accuracy", 0.0),
                    "unique_contributions": data.get("unique_contributions", {}),
                    "num_unique_samples": data.get("num_unique_samples", 0),
                }
            elif "control" in file_name:
                if N not in results_per_benchmark[benchmark_dir]["control"]:
                    results_per_benchmark[benchmark_dir]["control"][N] = {}
                results_per_benchmark[benchmark_dir]["control"][N] = {
                    "best_of_N_accuracy": data.get("best_of_N_accuracy", 0.0),
                    "majority_vote_accuracy": data.get("majority_vote_accuracy", 0.0),
                    "unique_contributions": data.get("unique_contributions", {}),
                    "num_unique_samples": data.get("num_unique_samples", 0),
                    "best_of_N_accuracy_std": data.get("best_of_N_accuracy_std", None),
                }
            elif "results_N3" in file_name:  # 3 experts case
                results_per_benchmark[benchmark_dir]["3_experts"] = {
                    "best_of_N_accuracy": data.get("best_of_N_accuracy", 0.0),
                    "majority_vote_accuracy": data.get("majority_vote_accuracy", 0.0),
                    "unique_contributions": data.get("unique_contributions", {}),
                    "num_unique_samples": data.get("num_unique_samples", 0),
                }

    logger.info(f"Got baseline results for benchmarks: {results_per_benchmark.keys()}")

    # Average over the benchmarks.
    average_over_control_N = {}
    first_benchmark_dir = list(results_per_benchmark.keys())[0]
    for N in results_per_benchmark[first_benchmark_dir]["control"]:
        average_over_control_N[N] = {
            "best_of_N_accuracy": sum(
                [
                    results_per_benchmark[benchmark_dir]["control"]
                    .get(N, {})
                    .get("best_of_N_accuracy", 0.0)
                    for benchmark_dir in results_per_benchmark
                ]
            )
            / len(results_per_benchmark),
            "majority_vote_accuracy": sum(
                [
                    results_per_benchmark[benchmark_dir]["control"]
                    .get(N, {})
                    .get("majority_vote_accuracy", 0.0)
                    for benchmark_dir in results_per_benchmark
                ]
            )
            / len(results_per_benchmark),
        }
        # Add stds
        average_over_control_N[N]["best_of_N_accuracy_std"] = sum(
            [
                results_per_benchmark[benchmark_dir]["control"]
                .get(N, {})
                .get("best_of_N_accuracy_std", 0.0)
                for benchmark_dir in results_per_benchmark
                if results_per_benchmark[benchmark_dir]["control"]
                .get(N, {})
                .get("best_of_N_accuracy_std", None)
                is not None
            ]
        ) / len(results_per_benchmark)
    averaged_across_benchmarks_results = {
        "big_model": {
            model_size: {
                "best_of_N_accuracy": sum(
                    [
                        results_per_benchmark[benchmark_dir]["big_model"]
                        .get(model_size, {})
                        .get("best_of_N_accuracy", 0.0)
                        for benchmark_dir in results_per_benchmark
                    ]
                )
                / len(results_per_benchmark),
                "majority_vote_accuracy": sum(
                    [
                        results_per_benchmark[benchmark_dir]["big_model"]
                        .get(model_size, {})
                        .get("majority_vote_accuracy", 0.0)
                        for benchmark_dir in results_per_benchmark
                    ]
                )
                / len(results_per_benchmark),
            }
            for model_size in results_per_benchmark[first_benchmark_dir]["big_model"]
        },
        "3_experts": {
            "best_of_N_accuracy": sum(
                [
                    results_per_benchmark[benchmark_dir]["3_experts"].get(
                        "best_of_N_accuracy", 0.0
                    )
                    for benchmark_dir in results_per_benchmark
                ]
            )
            / len(results_per_benchmark),
            "majority_vote_accuracy": sum(
                [
                    results_per_benchmark[benchmark_dir]["3_experts"].get(
                        "majority_vote_accuracy", 0.0
                    )
                    for benchmark_dir in results_per_benchmark
                ]
            )
            / len(results_per_benchmark),
        },
        "control": average_over_control_N,
    }

    return results_per_benchmark, averaged_across_benchmarks_results


def get_best_of_N_results_per_benchmark_per_selection_method(
    best_of_N_results_dir_path: str,
    relevant_selection_methods: list[str],
    relevant_benchmarks: list[str] = None,
) -> dict:
    """
    Get the best-of-N results per benchmark per selection method.
    The best-of-N results directory has the following structure:
    <best_of_N_results_dir_path>/
    ├── <benchmark_name>/
    │   ├── <selection_method>/
    │   │   ├── results_N1.json
    │   │   ├── results_N2.json
    │   │   ├── ...
    │   ├── <selection_method>/
    │   │   ├── results_N1.json
    │   │   ├── results_N2.json
    │   │   ├── ...
    ├── ...

    Returns:
        dict[str, dict[str, dict[str, dict[str, float]]]]: A dictionary of dictionaries containing the best-of-N results per benchmark per selection method.
        In the format of:
        {
            <benchmark_name>: {
                <selection_method>: {
                    <N>: {
                        "best_of_N_accuracy": float,
                        "majority_vote_accuracy": float,
                        "unique_contributions": {
                            <model_name>: int,
                            ...
                        },
                        "num_unique_samples": int,
                    },
                    <N+1>: {
                        "best_of_N_accuracy": float,
                        "majority_vote_accuracy": float,
                        "unique_contributions": {
                            <model_name>: int,
                            ...
                        },
                        "num_unique_samples": int,
                    },
                    ...
                },
                ...
            },
            ...
        }
        Average over the benchmarks.
        {
            <selection_method>: {
                <N>: {
                    "best_of_N_accuracy": float,
                    "majority_vote_accuracy": float,
                },
                ...
            },
        }
    """
    results_per_benchmark_per_selection_method = {}

    # Average over the benchmarks.
    averaged_across_benchmarks_results = {}

    for selection_method in relevant_selection_methods:
        averaged_across_benchmarks_results[selection_method] = {
            N: {
                "best_of_N_accuracy": [],
                "majority_vote_accuracy": [],
            }
            for N in range(1, 9)
        }

    print(
        f"Getting best-of-N results per benchmark per selection method from {best_of_N_results_dir_path}."
    )

    # Iterate through each benchmark directory
    for benchmark_dir in os.listdir(best_of_N_results_dir_path):
        benchmark_path = os.path.join(best_of_N_results_dir_path, benchmark_dir)
        if not os.path.isdir(benchmark_path) or (
            relevant_benchmarks is not None and benchmark_dir not in relevant_benchmarks
        ):
            continue

        results_per_benchmark_per_selection_method[benchmark_dir] = {}

        # Iterate through each selection method
        for selection_method in relevant_selection_methods:
            selection_method_path = os.path.join(benchmark_path, selection_method)
            if not os.path.isdir(selection_method_path):
                # Remove selection method from relevant_selection_methods
                relevant_selection_methods.remove(selection_method)
                logger.warning(
                    f"Selection method {selection_method} not found for benchmark {benchmark_dir}."
                )
                continue

            results_per_benchmark_per_selection_method[benchmark_dir][
                selection_method
            ] = {}

            # Process all JSON files in the selection method directory
            for file_name in os.listdir(selection_method_path):
                if not file_name.endswith(".json"):
                    continue

                file_path = os.path.join(selection_method_path, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Extract N from filename (e.g., results_N3.json -> 3)
                N = int(file_name.split("_N")[-1].split(".")[0])

                # Results for one benchmark.
                results_per_benchmark_per_selection_method[benchmark_dir][
                    selection_method
                ][N] = {
                    "best_of_N_accuracy": data.get("best_of_N_accuracy", 0.0),
                    "majority_vote_accuracy": data.get("majority_vote_accuracy", 0.0),
                    "unique_contributions": data.get("unique_contributions", {}),
                    "num_unique_samples": data.get("num_unique_samples", 0),
                }

                # Average over the benchmarks.
                averaged_across_benchmarks_results[selection_method][N][
                    "best_of_N_accuracy"
                ].append(data.get("best_of_N_accuracy", 0.0))
                averaged_across_benchmarks_results[selection_method][N][
                    "majority_vote_accuracy"
                ].append(data.get("majority_vote_accuracy", 0.0))

    # Average over the benchmarks.
    for selection_method in averaged_across_benchmarks_results:
        for N in averaged_across_benchmarks_results[selection_method]:
            averaged_across_benchmarks_results[selection_method][N][
                "best_of_N_accuracy"
            ] = sum(
                averaged_across_benchmarks_results[selection_method][N][
                    "best_of_N_accuracy"
                ]
            ) / len(
                averaged_across_benchmarks_results[selection_method][N][
                    "best_of_N_accuracy"
                ]
            )
            averaged_across_benchmarks_results[selection_method][N][
                "majority_vote_accuracy"
            ] = sum(
                averaged_across_benchmarks_results[selection_method][N][
                    "majority_vote_accuracy"
                ]
            ) / len(
                averaged_across_benchmarks_results[selection_method][N][
                    "majority_vote_accuracy"
                ]
            )

    return (
        results_per_benchmark_per_selection_method,
        averaged_across_benchmarks_results,
    )


def plot_best_of_N_results_per_benchmark(
    best_of_N_results_for_one_benchmark_per_selection_method: dict,
    baselines_results: dict,
    experiment_path: str,
    benchmark: str,
):
    """
    Plot the best-of-N results for one benchmark per selection method.
    Saves the plot to the experiment directory under eval/best_of_N_plots.
    """
    plt.figure(figsize=(10, 6))

    # Plot the best-of-N results per selection method
    for (
        selection_method,
        results,
    ) in best_of_N_results_for_one_benchmark_per_selection_method.items():
        N_values = sorted(results.keys())
        accuracies = [results[N]["best_of_N_accuracy"] for N in N_values]
        plt.plot(N_values, accuracies, "o-", label=selection_method)

    # Plot the baselines
    # Plot big_model if available
    if "big_model" in baselines_results and baselines_results["big_model"]:
        for i, model_size in enumerate(baselines_results["big_model"]):
            plt.axhline(
                y=baselines_results["big_model"][model_size]["best_of_N_accuracy"],
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                linestyle="--",
                label=f"Big Model {model_size}B",
            )

    # Plot 3_experts if available
    if "3_experts" in baselines_results and baselines_results["3_experts"]:
        plt.axhline(
            y=baselines_results["3_experts"]["best_of_N_accuracy"],
            color="gray",
            linestyle=":",
            label="3 Experts",
        )

    # Plot control results if available
    if "control" in baselines_results and baselines_results["control"]:
        control_N_values = sorted(baselines_results["control"].keys())
        control_accuracies = [
            baselines_results["control"][N]["best_of_N_accuracy"]
            for N in control_N_values
        ]
        # Gather stds, defaulting to 0 if not present
        control_stds = [
            baselines_results["control"][N].get("best_of_N_accuracy_std", 0)
            for N in control_N_values
        ]
        # Check if any std is not None and not 0
        if any(std is not None and std != 0 for std in control_stds):
            # Replace None with 0 for errorbar
            control_stds = [std if std is not None else 0 for std in control_stds]
            plt.errorbar(
                control_N_values,
                control_accuracies,
                yerr=control_stds,
                fmt="s--",
                color="gray",
                label="Control",
                capsize=4,
            )
        else:
            plt.plot(
                control_N_values,
                control_accuracies,
                "s--",
                color="gray",
                label="Control",
            )

    plt.xlabel("Number of Models (N)")
    plt.ylabel("Best-of-N Accuracy")
    plt.title(f"Best-of-N Results for {benchmark}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(experiment_path, "eval", "best_of_N_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{benchmark}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def create_main_best_of_N_plots(
    experiment_path: str,
    baselines_results_dir_path: str,
    relevant_selection_methods: list[str],
):
    """
    Create the main best-of-N plots for the experiment.

    Args:
        experiment_path: Path to the experiment directory.
        baselines_results_dir_path: Path to the baselines directory.
        relevant_selection_methods: List of selection methods to plot.
    """

    best_of_N_results_dir_path = os.path.join(experiment_path, "eval", "best_of_N")

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

    (
        baselines_results_per_benchmark,
        baselines_averaged_across_benchmarks_results,
    ) = get_baselines_results(baselines_results_dir_path, relevant_benchmarks)

    # Plot the best-of-N results per benchmark per selection method.
    for benchmark in best_of_N_results_per_benchmark_per_selection_method.keys():
        plot_best_of_N_results_per_benchmark(
            best_of_N_results_per_benchmark_per_selection_method[benchmark],
            baselines_results_per_benchmark[benchmark],
            experiment_path,
            benchmark,
        )

    # Plot the averaged across benchmarks results.
    plot_best_of_N_results_per_benchmark(
        averaged_across_benchmarks_results,
        baselines_averaged_across_benchmarks_results,
        experiment_path,
        "averaged_across_benchmarks",
    )

    pass


def main():
    experiment_path = "outputs/2025-05-27/08-31-08"
    baselines_results_dir_path = "visualization/baselines/zero_shot/best_of_N"
    relevant_selection_methods = [
        "local_dataset_based_fitness_across_entire_archive",
        "local_dataset_based_fitness_one_per_gen",
    ]
    create_main_best_of_N_plots(
        experiment_path, baselines_results_dir_path, relevant_selection_methods
    )


if __name__ == "__main__":
    main()
