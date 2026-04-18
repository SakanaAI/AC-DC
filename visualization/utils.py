import os
import json
import glob
import logging
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.manifold import TSNE
import networkx as nx
from sklearn.cluster import HDBSCAN
import plotly.graph_objects as go
import plotly.express as px
import imageio
from PIL import Image
import io
import re
import networkx as nx
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLOR_PALETTE = [
    "#0068c9",  # Blue
    "#83c9ff",  # Light Blue
    "#7defa1",  # light green
    "#ffabab",  # Peach
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#bcbd22",  # yellow-green
    "#ff7f0e",  # orange
    # "#ff2b2b",  # Red
    "#17becf",  # cyan
    "#98df8a",  # light green
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#6AD6F7",
    "#E0F76A",
    "#8F6AF7",
    "#F76A6A",
    "#6AF7F7",
    "#F76A9A",
    "#F76A6A",
    "#6AF7F7",
    "#F76A6A",
    "#BD6AF7",
    "#F7C66A",
    "#F7B26A",
    "#FAE8C7",
    "#6ADCF7",
    "#7f7f7f",  # gray
    "#d62728",  # Red
]


def get_unique_contributions_relative_to_all_samples(
    gen_results: dict,
):
    """Get the unique contributions relative to all samples."""
    all_unique_contributions = sum(gen_results["unique_contributions"].values())
    return (all_unique_contributions / gen_results["num_unique_samples"]) * 100


def extract_generation(model_name: str) -> int:
    """Extract generation number from model name like 'gen_X_ind_Y'."""
    match = re.search(r"gen_(\d+)", model_name)
    if match:
        return int(match.group(1))
    return 0  # Seed models have generation 0


def multi_word_string_to_readable_string(multi_word_string: str) -> str:
    """Convert a multi-word string to a readable string."""
    # Split by _ and capitalize the first letter of each word
    return " ".join(word.capitalize() for word in multi_word_string.split("_"))


def cast_list_elements_to_dtype(list_of_elements: list, dtype: type) -> list:
    """Cast the elements of a list to a given dtype."""
    return [dtype(element) for element in list_of_elements]


def mix_colors(colors_to_mix: list[str]) -> str:
    """
    Get a mixed color from a list of colors.

    Args:
        colors_to_mix: list of colors to mix. E.g. ["#000000", "#FFFFFF"]

    Returns:
        str: a mixed color. E.g. "#808080"
    """
    if not colors_to_mix:
        raise ValueError("Cannot mix an empty list of colors")

    if len(colors_to_mix) == 1:
        return colors_to_mix[0]

    total_r = 0
    total_g = 0
    total_b = 0

    for color in colors_to_mix:
        # Remove the '#' and convert hex to int
        hex_color = color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color format: {color}")

        # Extract RGB components
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        total_r += r
        total_g += g
        total_b += b

    # Calculate average RGB values
    num_colors = len(colors_to_mix)
    avg_r = round(total_r / num_colors)
    avg_g = round(total_g / num_colors)
    avg_b = round(total_b / num_colors)

    # Convert back to hex format
    mixed_color = f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"
    return mixed_color


def compute_and_load_synth_task_to_tSNE_mapping(
    experiment_path: str,
):
    """
    Load the task embeddings from the experiment path.
    Compute the tSNE for the task embeddings.
    Save the tSNE mapping to a json file.
    """

    path_to_tasks = os.path.join(experiment_path, "generated_tasks", "pool")
    path_to_vector_db = os.path.join(experiment_path, "vector_db_historical")

    # See if tasks_tSNSE.json exists in experiment_path/generated_tasks/pool
    tasks_tSNE_path = os.path.join(path_to_tasks, "tasks_tSNE.json")
    tasks_hdbscan_path = os.path.join(path_to_tasks, "tasks_hdbscan.json")
    tasks_tSNE_mapping = None
    tasks_hdbscan_mapping = None
    if os.path.exists(tasks_tSNE_path):
        try:
            with open(tasks_tSNE_path, "r") as f:
                tasks_tSNE_mapping = json.load(f)
            # turn the values into float lists
            tasks_tSNE_mapping = {
                k: cast_list_elements_to_dtype(v, dtype=float)
                for k, v in tasks_tSNE_mapping.items()
            }
        except json.JSONDecodeError:
            logger.warning(
                f"tasks_tSNE.json is not a valid json file. Deleting it and recomputing..."
            )
            os.remove(tasks_tSNE_path)

    if os.path.exists(tasks_hdbscan_path):
        try:
            with open(tasks_hdbscan_path, "r") as f:
                tasks_hdbscan_mapping = json.load(f)
            # turn the values into ints
            tasks_hdbscan_mapping = {
                k: int(v) for k, v in tasks_hdbscan_mapping.items()
            }
        except json.JSONDecodeError:
            logger.warning(
                f"tasks_hdbscan.json is not a valid json file. Deleting it and recomputing..."
            )
            os.remove(tasks_hdbscan_path)

    if tasks_tSNE_mapping is not None and tasks_hdbscan_mapping is not None:
        return tasks_tSNE_mapping, tasks_hdbscan_mapping
    else:
        if tasks_tSNE_mapping is None:
            logger.info("Computing tSNE for the tasks...")
        if tasks_hdbscan_mapping is None:
            logger.info("Computing HDBSCAN for the tasks...")

    task_embeddings_tSNE = None
    if tasks_tSNE_mapping is None:
        # Get all active_pool_gen_X.json files in experiment_path/generated_tasks/pool
        active_pool_files = glob.glob(
            os.path.join(
                path_to_tasks,
                "active_pool_gen_*.json",
            )
        )

        # Get the task names from the active_pool_files
        task_names = set()
        for active_pool_file in active_pool_files:
            with open(active_pool_file, "r") as f:
                active_pool_tasks = json.load(f)
            for task_dir_path in active_pool_tasks:
                task_name = task_dir_path.split("/")[-1]
                task_names.add(task_name)

        # Load the task embeddings from the vector database in the vectors dir
        task_name_to_embedding_vector = {}
        for task_name in task_names:
            task_name_to_embedding_vector[task_name] = np.load(
                os.path.join(path_to_vector_db, "vectors", task_name + ".npy")
            )

        # sort the task_name_to_embedding_vector based on the task_name
        task_name_to_embedding_vector = dict(
            sorted(
                task_name_to_embedding_vector.items(),
                key=lambda x: int(x[0].split("_")[1]),
            )
        )

        # Compute the tSNE for the task embeddings
        # Parameters used by ACD
        # n_components 2
        # perplexity 50
        # learning_rate 200
        # n_iter 3000
        # init pca
        # random_state 42
        # early_exaggeration 6.0
        tsne = TSNE(
            n_components=2,
            perplexity=50,
            learning_rate=200,
            n_iter=3000,
            init="pca",
            random_state=42,
            early_exaggeration=6.0,
        )
        task_embeddings_tSNE = tsne.fit_transform(
            np.array(list(task_name_to_embedding_vector.values()))
        )
        tasks_tSNE_mapping = {
            task_name: cast_list_elements_to_dtype(
                list(task_embeddings_tSNE[i]), dtype=float
            )
            for i, task_name in enumerate(task_name_to_embedding_vector.keys())
        }

    # Get clusters using hdbscan
    if tasks_hdbscan_mapping is None:
        if task_embeddings_tSNE is None:
            task_embeddings_tSNE = np.array(list(tasks_tSNE_mapping.values()))

        ### Parameters used by ACD
        # min_cluster_size=16
        # min_samples=4
        # cluster_selection_epsilon=2
        # cluster_selection_method="eom"
        # metric="euclidean"

        hdbscan_clusters = HDBSCAN(min_cluster_size=10).fit(
            task_embeddings_tSNE
        )
        tasks_hdbscan_mapping = {
            task_name: int(hdbscan_clusters.labels_[i])
            for i, task_name in enumerate(tasks_tSNE_mapping.keys())
        }

    # Save the tSNE mapping to a json file
    try:
        if not os.path.exists(tasks_tSNE_path):
            with open(tasks_tSNE_path, "w") as f:
                json.dump(tasks_tSNE_mapping, f)
    except Exception as e:
        logger.error(f"Error saving tSNE mapping to {tasks_tSNE_path}: {e}")

    try:
        if not os.path.exists(tasks_hdbscan_path):
            with open(tasks_hdbscan_path, "w") as f:
                json.dump(tasks_hdbscan_mapping, f)
    except Exception as e:
        logger.error(
            f"Error saving HDBSCAN mapping to {tasks_hdbscan_path}: {e}"
        )

    return tasks_tSNE_mapping, tasks_hdbscan_mapping


def get_task_tSNE_graph_object(
    task_name_to_tSNE_embedding: dict,
):
    """Get the tSNE graph object for a task."""
    G = nx.Graph()
    for task_name, embedding in task_name_to_tSNE_embedding.items():
        G.add_node(task_name, embedding=embedding)
    return G


def get_task_info(experiment_path: str, task_name: str):
    """Get the task info for a task. Get a tasks `task.json` file.
    The task info is a dictionary with the following keys:
    - "name_of_task":
    - "description_of_task":
    - "capability_being_measured":
    - "estimated_human_difficulty":
    - "example_instruction":
    """
    path_to_tasks = os.path.join(experiment_path, "generated_tasks", "pool")

    # Get the task info from the task_info.json file
    task_info_path = os.path.join(path_to_tasks, task_name, "task.json")
    with open(task_info_path, "r") as f:
        task_info = json.load(f)
    return task_info


def build_task_hover_text(
    experiment_path: str,
    task_name: str,
    solved_by_models: list[str] = None,
):
    """Build the hover text for a task."""
    task_info = get_task_info(experiment_path, task_name)
    hover_text = f"<br><b>Task Name:</b> {task_info['name_of_task']}<br>"
    hover_text += f"<b>Description:</b> {task_info['description_of_task']}<br>"
    hover_text += f"<b>Capability Being Measured:</b> {task_info['capability_being_measured']}<br>"
    hover_text += f"<b>Estimated Human Difficulty:</b> {task_info['estimated_human_difficulty']}<br>"

    # Format the example to have a max of 100 characters per new line
    new_example_instruction = ""
    all_words = task_info["example_instruction"].split(" ")
    words_per_line = 15
    for i in range(0, len(all_words), words_per_line):
        end = min(i + words_per_line, len(all_words))
        new_example_instruction += " ".join(all_words[i:end]) + "<br>"
    example_instruction = new_example_instruction

    hover_text += f"<b>Example Instruction:</b> {example_instruction}<br>"
    if solved_by_models:
        hover_text += f"<b>Solved by:</b> {solved_by_models}"
    return hover_text


def get_single_answer_from_pop_data(
    single_answer_from_pop_dir: str,
    existing_benchmarks: list[str],
    is_baselines: bool = False,
):
    """Get the single answer from pop data.

    Args:
        single_answer_from_pop_dir: The directory containing the single answer from pop data.
        existing_benchmarks: The list of benchmarks to get the single answer from pop data for.

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
    single_answer_from_pop_data = {}

    benchmarks_in_single_answer_from_pop_dir = glob.glob(
        os.path.join(single_answer_from_pop_dir, "*")
    )

    # Get all task force selection methods
    task_force_selection_methods = []
    # check if there is a dir that is a benchmark
    legacy = False
    # if there is a json file in the first benchmark dir, then it is legacy
    files_in_benchmark_dir = glob.glob(
        os.path.join(benchmarks_in_single_answer_from_pop_dir[0], "*.json")
    )
    # if len(files_in_benchmark_dir) > 0 or is_baselines:
    if len(files_in_benchmark_dir) > 0:
        legacy = True
        logger.warning(
            "The SAS data is in the legacy format or is baselines data. "
            "Assuming the models used are from gsv coverage."
        )
        task_force_selection_methods = ["global_skill_vector_coverage"]
    else:
        task_force_selection_methods = [
            os.path.basename(task_force_selection_method_dir)
            for task_force_selection_method_dir in glob.glob(
                os.path.join(benchmarks_in_single_answer_from_pop_dir[0], "*")
            )
        ]

    # if legacy or is_baselines:
    #     logger.warning(
    #         "The SAS data is in the legacy format or is baselines data. "
    #         "Assuming the models used are from gsv coverage."
    #     )
    #     task_force_selection_methods = ["global_skill_vector_coverage"]
    # else:
    #     task_force_selection_methods = [
    #         os.path.basename(dir)
    #         for dir in dirs_in_single_answer_from_pop_dir
    #         if os.path.isdir(dir)
    #     ]

    # Loop over all task force selection methods
    for task_force_selection_method in task_force_selection_methods:
        if task_force_selection_method not in single_answer_from_pop_data:
            single_answer_from_pop_data[task_force_selection_method] = {}

        # Loop over all relevant benchmarks
        for benchmark in existing_benchmarks:
            if legacy:
                path_to_data_dir = os.path.join(
                    single_answer_from_pop_dir, benchmark
                )
            else:
                path_to_data_dir = os.path.join(
                    single_answer_from_pop_dir,
                    benchmark,
                    task_force_selection_method,
                )

            if not os.path.exists(path_to_data_dir):
                continue

            if (
                benchmark
                not in single_answer_from_pop_data[task_force_selection_method]
            ):
                single_answer_from_pop_data[task_force_selection_method][
                    benchmark
                ] = {}

            # Loop over all model groups
            for model_group_name_path in glob.glob(
                os.path.join(path_to_data_dir, "results_*.json")
            ):
                with open(model_group_name_path, "r") as f:
                    results_model_group = json.load(f)

                # Get the model group name
                model_group_name = (
                    os.path.basename(model_group_name_path)
                    .replace("results_", "")
                    .split(".")[0]
                )

                if (
                    model_group_name
                    not in single_answer_from_pop_data[
                        task_force_selection_method
                    ][benchmark]
                ):
                    single_answer_from_pop_data[task_force_selection_method][
                        benchmark
                    ][model_group_name] = {}

                # get all results, i.e., all keys that don't start with "model_distribution"
                all_results = {
                    k: v
                    for k, v in results_model_group.items()
                    if not k.startswith("model_distribution")
                }
                single_answer_from_pop_data[task_force_selection_method][
                    benchmark
                ][model_group_name]["results"] = all_results

                # get all model distributions, i.e., all keys that start with "model_distribution"
                all_model_distributions = {
                    k.replace("model_distribution_", ""): v
                    for k, v in results_model_group.items()
                    if k.startswith("model_distribution")
                }

                all_models = set(
                    [
                        model_name
                        for distribution_value in all_model_distributions.values()
                        for model_name in distribution_value.keys()
                    ]
                )

                # Add the distribution percentage per model for each selection method
                if (
                    "model_distribution"
                    not in single_answer_from_pop_data[
                        task_force_selection_method
                    ][benchmark][model_group_name]
                ):
                    single_answer_from_pop_data[task_force_selection_method][
                        benchmark
                    ][model_group_name]["model_distribution"] = {}

                for model in all_models:
                    if (
                        model
                        not in single_answer_from_pop_data[
                            task_force_selection_method
                        ][benchmark][model_group_name]["model_distribution"]
                    ):
                        single_answer_from_pop_data[
                            task_force_selection_method
                        ][benchmark][model_group_name]["model_distribution"][
                            model
                        ] = {}

                    for selection_method in all_model_distributions.keys():
                        model_distribution = all_model_distributions[
                            selection_method
                        ]
                        single_answer_from_pop_data[
                            task_force_selection_method
                        ][benchmark][model_group_name]["model_distribution"][
                            model
                        ][
                            selection_method
                        ] = model_distribution.get(
                            model, 0
                        )

    # Dirty fix for baseline data
    ## We don't want to have a "task_force_selection_method" key in the output dir
    if is_baselines:
        single_answer_from_pop_data = single_answer_from_pop_data[
            "global_skill_vector_coverage"
        ]

    return single_answer_from_pop_data


def shorten_model_name(model_name: str) -> str:
    """Shorten the model name."""
    if "coding" in model_name.lower() or "coder" in model_name.lower():
        return "Code-Expert"
    if "gsm8k" in model_name.lower():
        return "Gsm8k-Expert"
    if "math" in model_name.lower():
        return "Math-Expert"
    if "seed" in model_name.lower():
        seed_number = model_name.split("_")[-1]
        return f"Instruct-{seed_number}"
    if "Instruct" in model_name:
        return "Instruct"
    return model_name


def load_generation_files(tasks_pool_dir: str) -> Dict[int, List[str]]:
    """
    Loads all active_pool_gen_x.json files and returns a mapping of generation number to task list.
    """
    generations = {}
    for entry in os.listdir(tasks_pool_dir):
        if entry.startswith("active_pool_gen_") and entry.endswith(".json"):
            gen_num = int(entry.split("_")[-1].split(".")[0])
            with open(os.path.join(tasks_pool_dir, entry), "r") as f:
                generations[gen_num] = json.load(f)
    return dict(sorted(generations.items()))


def get_adaptation_generations_and_gen_step_size(
    tasks_pool_per_generations: Dict[int, List[str]],
) -> List[int]:
    """
    Gets the generations of the adaptation tasks.
    """
    adaptation_gens = []
    current_active_tasks = set()
    for gen, tasks in tasks_pool_per_generations.items():
        if current_active_tasks != set(tasks):
            current_active_tasks = set(tasks)
            adaptation_gens.append(gen)

    gen_step_size = adaptation_gens[2] - adaptation_gens[1]

    return adaptation_gens, gen_step_size


def get_global_min_max_values(
    task_name_to_tSNE_embedding: Dict[str, np.ndarray],
    all_tasks_graph: nx.Graph = None,
) -> Tuple[float, float, float, float]:
    """
    Gets the global min/max values from all embeddings to fix axes.
    """
    # Calculate global min/max values from all embeddings to fix axes
    all_x_values = []
    all_y_values = []

    if all_tasks_graph:
        # Get all embeddings from the graph
        for node in all_tasks_graph.nodes():
            embedding = all_tasks_graph.nodes[node]["embedding"]
            all_x_values.append(embedding[0])
            all_y_values.append(embedding[1])
    else:
        # Get all embeddings from the dictionary
        for embedding in task_name_to_tSNE_embedding.values():
            all_x_values.append(embedding[0])
            all_y_values.append(embedding[1])

    # Calculate global bounds
    global_x_min, global_x_max = min(all_x_values), max(all_x_values)
    global_y_min, global_y_max = min(all_y_values), max(all_y_values)

    # Add some padding to the bounds
    x_padding = (global_x_max - global_x_min) * 0.05
    y_padding = (global_y_max - global_y_min) * 0.05
    global_x_min -= x_padding
    global_x_max += x_padding
    global_y_min -= y_padding
    global_y_max += y_padding

    return global_x_min, global_x_max, global_y_min, global_y_max


def create_task_embedding_plot_for_generation(
    task_name_to_tSNE_embedding: Dict[str, np.ndarray],
    task_name_to_hdbscan_cluster: Dict[str, int],
    experiment_path: str,
    relevant_gen: int,
    all_tasks_graph: nx.Graph = None,
    x_axis_range: Tuple[float, float] = None,
    y_axis_range: Tuple[float, float] = None,
) -> go.Figure:
    """
    Creates a plotly figure for task embeddings for a specific generation.
    Returns the figure object instead of displaying it.
    """
    tasks_pool_dir = os.path.join(experiment_path, "generated_tasks", "pool")

    # Handle "All" generations case (represented by -1)
    if relevant_gen == -1:
        title = "tSNE Plot for All Tasks"
        # Use all tasks from the graph
        relevant_task_names = (
            list(all_tasks_graph.nodes())
            if all_tasks_graph
            else list(task_name_to_tSNE_embedding.keys())
        )
    else:
        generations = load_generation_files(tasks_pool_dir)

        # get the task names for the relevant generation
        task_paths = generations[relevant_gen]
        task_names = [task_path.split("/")[-1] for task_path in task_paths]
        title = f"tSNE Plot for Generation {relevant_gen}"
        relevant_task_names = task_names

    ### Create plotly chart ############################################
    # Create a plotly chart with the tSNE plot object
    # For each node, add a hover text with the task name and the models that have solved it
    fig = go.Figure()
    x_values = []
    y_values = []
    colors = []
    hovertext = []

    # Use the all_tasks_graph if provided, otherwise create a filtered graph
    num_clusters = len(set(task_name_to_hdbscan_cluster.values()))
    if num_clusters > len(COLOR_PALETTE):
        logger.error(
            f"Number of clusters ({num_clusters}) is greater than the number of colors ({len(COLOR_PALETTE)}). "
            "Please increase the number of colors or decrease the number of clusters."
        )
        return fig

    if all_tasks_graph:
        # Filter the graph to only show relevant tasks
        for task_name in relevant_task_names:
            if task_name in all_tasks_graph.nodes():
                embedding = all_tasks_graph.nodes[task_name]["embedding"]
                x_values.append(embedding[0])
                y_values.append(embedding[1])
                hovertext.append(
                    build_task_hover_text(
                        experiment_path=experiment_path,
                        task_name=task_name,
                    )
                )
                colors.append(
                    COLOR_PALETTE[task_name_to_hdbscan_cluster[task_name]]
                )
    else:
        # Fallback to recomputing/reloading the tSNE embeddings if no graph provided
        filtered_embeddings = {
            task_name: task_name_to_tSNE_embedding[task_name]
            for task_name in relevant_task_names
            if task_name in task_name_to_tSNE_embedding
        }
        G = get_task_tSNE_graph_object(filtered_embeddings)

        for task_name in G.nodes():
            x_values.append(G.nodes[task_name]["embedding"][0])
            y_values.append(G.nodes[task_name]["embedding"][1])
            hovertext.append(
                build_task_hover_text(
                    experiment_path=experiment_path,
                    task_name=task_name,
                )
            )
            colors.append(
                COLOR_PALETTE[task_name_to_hdbscan_cluster[task_name]]
            )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(
                color=colors,
                size=10,
            ),
            hovertext=hovertext,
            opacity=0.7,
        )
    )

    # Set fixed axis ranges
    if x_axis_range is None or y_axis_range is None:
        global_x_min, global_x_max, global_y_min, global_y_max = (
            get_global_min_max_values(
                task_name_to_tSNE_embedding, all_tasks_graph
            )
        )
    else:
        global_x_min, global_x_max = x_axis_range
        global_y_min, global_y_max = y_axis_range

    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[global_x_min, global_x_max], title="tSNE Dimension 1"
        ),
        yaxis=dict(
            range=[global_y_min, global_y_max], title="tSNE Dimension 2"
        ),
        width=800,
        height=600,
    )

    return fig, title


def create_task_embedding_gif(
    experiment_path: str,
    output_path: str = None,
    fps: int = 2,
    include_all_generation: bool = True,
) -> str:
    """
    Creates a GIF showing the evolution of task embeddings across generations.

    Args:
        experiment_path: Path to the experiment directory
        output_path: Path where to save the GIF. If None, saves in experiment_path
        fps: Frames per second for the GIF
        include_all_generation: Whether to include a frame showing all generations

    Returns:
        str: Path to the created GIF file
    """
    logger.info("Starting to create task embedding GIF...")

    # Set output path
    if output_path is None:
        output_path = os.path.join(
            experiment_path, "images", "task_embedding_evolution.gif"
        )

    # Load tSNE embeddings and clusters
    task_name_to_tSNE_embedding, task_name_to_hdbscan_cluster = (
        compute_and_load_synth_task_to_tSNE_mapping(experiment_path)
    )

    # Create the graph object with all embeddings
    all_tasks_graph = get_task_tSNE_graph_object(task_name_to_tSNE_embedding)

    # Get global axis ranges for consistent scaling
    global_x_min, global_x_max, global_y_min, global_y_max = (
        get_global_min_max_values(task_name_to_tSNE_embedding, all_tasks_graph)
    )
    x_axis_range = (global_x_min, global_x_max)
    y_axis_range = (global_y_min, global_y_max)

    # Load generation files
    tasks_pool_dir = os.path.join(experiment_path, "generated_tasks", "pool")
    adaptation_gens, gen_step_size = (
        get_adaptation_generations_and_gen_step_size(
            load_generation_files(tasks_pool_dir)
        )
    )

    if not adaptation_gens:
        logger.error("No generation files found!")
        return None

    # Get all generation numbers and sort them
    logger.info(f"Found generations: {adaptation_gens}")

    images = []

    # Create plots for each generation
    for gen_num in adaptation_gens:
        logger.info(f"Creating plot for generation {gen_num}...")

        # Create the plot for this generation
        fig, _ = create_task_embedding_plot_for_generation(
            task_name_to_tSNE_embedding=task_name_to_tSNE_embedding,
            task_name_to_hdbscan_cluster=task_name_to_hdbscan_cluster,
            experiment_path=experiment_path,
            relevant_gen=gen_num,
            all_tasks_graph=all_tasks_graph,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
        )

        # Convert plotly figure to image
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img = Image.open(io.BytesIO(img_bytes))

        # Save to temporary file
        os.makedirs(
            os.path.join(experiment_path, "images", "embeddings_space_per_gen"),
            exist_ok=True,
        )
        temp_img_path = os.path.join(
            experiment_path,
            "images",
            "embeddings_space_per_gen",
            f"gen_{gen_num:03d}.png",
        )
        img.save(temp_img_path)
        images.append(temp_img_path)

    # Add "all generations" frame if requested
    if include_all_generation:
        logger.info("Creating plot for all generations...")
        fig_all, _ = create_task_embedding_plot_for_generation(
            task_name_to_tSNE_embedding=task_name_to_tSNE_embedding,
            task_name_to_hdbscan_cluster=task_name_to_hdbscan_cluster,
            experiment_path=experiment_path,
            relevant_gen=-1,  # -1 represents all generations
            all_tasks_graph=all_tasks_graph,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
        )

        img_bytes = fig_all.to_image(format="png", width=800, height=600)
        img = Image.open(io.BytesIO(img_bytes))

        temp_img_path = os.path.join(
            experiment_path,
            "images",
            "embeddings_space_per_gen",
            "gen_all.png",
        )
        img.save(temp_img_path)
        images.append(temp_img_path)

    # Create GIF from all images
    logger.info(f"Creating GIF from {len(images)} frames...")
    gif_images = []
    for img_path in images:
        gif_images.append(imageio.imread(img_path))

    # Save the GIF
    imageio.mimsave(output_path, gif_images, fps=fps)

    logger.info(f"GIF saved to: {output_path}")
    return output_path


def compute_fitness_from_skill_vector(skill_vector: np.ndarray) -> float:
    """Compute fitness from skill vector."""
    return round(np.mean(np.array(skill_vector)), 2)


def create_matplotlib_mapper(values, colormap="viridis"):
    """
    Create a color mapper using matplotlib's colormaps.

    Args:
        values: List of values to map
        colormap: Name of matplotlib colormap ('viridis', 'plasma', 'hot', 'cool', etc.)

    Returns:
        Function that maps values to colors
    """
    vmin, vmax = min(values), max(values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(colormap)

    def get_color(value, format="hex"):
        """Get color for a value"""
        normalized = norm(value)
        rgba = cmap(normalized)

        if format == "hex":
            return mcolors.to_hex(rgba)
        elif format == "rgb":
            return tuple(int(255 * c) for c in rgba[:3])
        elif format == "rgba":
            return rgba
        else:
            return rgba

    return get_color


def create_equally_spaced_positions(
    G: nx.DiGraph,
) -> Dict[str, Tuple[float, float]]:
    """Create equally spaced positions for the nodes in the graph."""
    pos = nx.spring_layout(G, k=2, iterations=50)
    # Adjust x-coordinates based on generation
    for node in G.nodes():
        gen = G.nodes[node].get("generation", 0)
        pos[node] = (
            gen,
            pos[node][1],
        )  # Use generation number as x-coordinate

    # For the nodes in each generation, space them out evenly
    nodes_by_generation = {}
    for node in G.nodes():
        gen = G.nodes[node].get("generation", 0)
        if gen not in nodes_by_generation:
            nodes_by_generation[gen] = []
        nodes_by_generation[gen].append(node)

    # Get the maximum and minimum y-coordinates for each generation
    y_coords_by_generation = {}
    for gen, nodes in nodes_by_generation.items():
        y_coords = [pos[node][1] for node in nodes]
        y_coords_by_generation[gen] = (min(y_coords), max(y_coords))

    # Calculate the spacing for each generation
    spacing_by_generation = {}
    for gen, (min_y, max_y) in y_coords_by_generation.items():
        n_nodes = len(nodes_by_generation[gen])
        spacing = (max_y - min_y) / (n_nodes - 1) if n_nodes > 1 else 0
        spacing_by_generation[gen] = spacing

    # Adjust the y-coordinates for each node in its generation
    for gen, nodes in nodes_by_generation.items():
        min_y = y_coords_by_generation[gen][0]
        spacing = spacing_by_generation[gen]
        for i, node in enumerate(nodes):
            pos[node] = (pos[node][0], min_y + i * spacing)

    return pos, nodes_by_generation


def create_interactive_generation_tree(
    G: nx.DiGraph,
    models_to_parent_models: Dict[str, List[str]],
    selected_model: str = None,
    top_N_model_names: Dict[str, np.ndarray] = {},
    positions: Dict[str, Tuple[float, float]] = None,
):
    """Create an interactive generation tree for models."""

    def is_seed_model(model_name: str) -> bool:
        """Check if model is a seed model (doesn't have gen_X pattern)."""
        # Check if model name contains "gen_" and a number
        return not re.search(r"gen_\d+", model_name)

    def get_linneage(
        model_name: str,
        models_to_parent_models: Dict[str, List[Tuple[str, float]]],
        lineage: dict[str, float] = {},
        model_fitness: float = None,
    ) -> set[str]:
        """Get the lineage of a model."""
        lineage[model_name] = model_fitness
        # break condition: if the model is a seed model
        if is_seed_model(model_name):
            return lineage

        # Get the parent models
        parent_models = models_to_parent_models.get(model_name, [])
        if len(parent_models) == 0:
            return lineage

        for parent_model, parent_model_fitness in parent_models:
            get_linneage(
                model_name=parent_model,
                models_to_parent_models=models_to_parent_models,
                lineage=lineage,
                model_fitness=parent_model_fitness,
            )

        return lineage

    # Use hierarchical layout with left-to-right orientation
    if positions is None:
        pos, _ = create_equally_spaced_positions(G)
    else:
        pos = positions

    ####################################################################

    # Find the path to highlight if a model is selected
    lineage = None
    get_heatmap_color = None
    if selected_model and selected_model in G.nodes():
        lineage = {}
        get_linneage(selected_model, models_to_parent_models, lineage)
        fitness_values = [
            value for value in list(lineage.values()) if value is not None
        ]
        get_heatmap_color = create_matplotlib_mapper(
            fitness_values, colormap="viridis"
        )
    elif top_N_model_names:
        fitness_values = list(top_N_model_names.values())
        get_heatmap_color = create_matplotlib_mapper(
            fitness_values, colormap="viridis"
        )

    # Create edge traces for highlighted and non-highlighted edges
    edge_x_highlighted = []
    edge_y_highlighted = []
    edge_x_other = []
    edge_y_other = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if lineage and edge[0] in lineage and edge[1] in lineage:
            edge_x_highlighted.extend([x0, x1, None])
            edge_y_highlighted.extend([y0, y1, None])
        else:
            edge_x_other.extend([x0, x1, None])
            edge_y_other.extend([y0, y1, None])

    # Create edge traces with different colors
    edge_trace_highlighted = go.Scatter(
        x=edge_x_highlighted,
        y=edge_y_highlighted,
        line=dict(width=2, color="red"),
        opacity=0.6,
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    edge_trace_other = go.Scatter(
        x=edge_x_other,
        y=edge_y_other,
        line=dict(width=0.8, color="#ccc"),
        opacity=0.3,
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    # Create node traces for highlighted and non-highlighted nodes
    node_x_highlighted = []
    node_y_highlighted = []
    node_text_highlighted = []
    node_labels_highlighted = []
    node_x_other = []
    node_y_other = []
    node_text_other = []
    node_labels_other = []
    node_colors = []
    # linneage_node_colors = []
    # node_colors_highlighted = []
    highlighted_node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_data = G.nodes[node]
        generation = node_data.get("generation", 0)
        is_seed = node_data.get("is_seed", False)
        parent_models = node_data.get("parent_models", [])

        # Create hover text
        if is_seed:
            hover_text = f"Seed Model: {node}<br>Generation: 0"
            # label = node
        else:
            hover_text = f"Model: {node}<br>Generation: {generation}"
            # label = node.split("_")[-1]  # Show just the individual index

        node_color = COLOR_PALETTE[9]

        if lineage and node in lineage:
            node_x_highlighted.append(x)
            node_y_highlighted.append(y)
            hover_text += f"<br>Fitness: {lineage[node]}"
            if parent_models:
                hover_text += f"<br>Parent Models: {', '.join(parent_models)}"
            node_text_highlighted.append(hover_text)
            node_name = shorten_model_name(node)
            node_labels_highlighted.append(node_name)
            node_fitness = lineage[node]
            if node_fitness is None:
                highlighted_node_colors.append("red")
            else:
                highlighted_node_colors.append(get_heatmap_color(node_fitness))
        elif top_N_model_names and node in top_N_model_names.keys():
            node_x_highlighted.append(x)
            node_y_highlighted.append(y)
            hover_text += f"<br>Fitness: {top_N_model_names[node]}"
            node_text_highlighted.append(hover_text)
            node_labels_highlighted.append(shorten_model_name(node))
            highlighted_node_colors.append(
                get_heatmap_color(top_N_model_names[node])
            )
        else:
            node_x_other.append(x)
            node_y_other.append(y)
            node_text_other.append(hover_text)
            node_labels_other.append(None)
            node_colors.append(node_color)

    # Create node traces with different colors if a specific model is selected
    if lineage or top_N_model_names:
        legend_title = "Local Fitness" if lineage else "Global Fitness"
        node_trace_highlighted = go.Scatter(
            x=node_x_highlighted,
            y=node_y_highlighted,
            mode="markers+text",
            hoverinfo="text",
            text=node_labels_highlighted,
            hovertext=node_text_highlighted,
            textposition="middle center",
            marker=dict(
                showscale=True,  # Enable colorbar
                color=highlighted_node_colors,
                colorscale="viridis",  # Use viridis colormap
                colorbar=dict(
                    title=legend_title,
                    x=1.1,  # Position colorbar to the right
                    len=0.8,  # Length of colorbar
                    thickness=20,  # Width of colorbar
                    outlinewidth=1,
                    outlinecolor="black",
                ),
                cmin=(
                    min(fitness_values) if fitness_values else 0
                ),  # Set colorbar range
                cmax=max(fitness_values) if fitness_values else 1,
                size=25,
                line_width=3,
            ),
            showlegend=True,
        )

        # Create node trace for non-highlighted nodes
        node_trace_other = go.Scatter(
            x=node_x_other,
            y=node_y_other,
            mode="markers+text",
            hoverinfo="text",
            text=node_labels_other,
            hovertext=node_text_other,
            textposition="middle center",
            marker=dict(
                showscale=False,
                color="grey",
                size=20,
                line_width=2,
                opacity=0.5,
            ),
            showlegend=False,
        )
        data = [
            edge_trace_other,
            edge_trace_highlighted,
            node_trace_other,
            node_trace_highlighted,
        ]
    # elif top_N_model_names:
    #     node_trace_highlighted = go.Scatter(
    #         x=node_x_highlighted,
    #         y=node_y_highlighted,
    #         mode="markers+text",
    #         hoverinfo="text",
    #         text=node_labels_highlighted,
    #         hovertext=node_text_highlighted,
    #         textposition="middle center",
    #         marker=dict(
    #             showscale=True,  # Enable colorbar
    #             color=highlighted_node_colors,
    #             colorscale="viridis",  # Use viridis colormap
    #             colorbar=dict(
    #                 title="Fitness",
    #                 x=1.1,  # Position colorbar to the right
    #                 len=0.8,  # Length of colorbar
    #                 thickness=20,  # Width of colorbar
    #                 outlinewidth=1,
    #                 outlinecolor="black",
    #             ),
    #             cmin=(
    #                 min(fitness_values) if fitness_values else 0
    #             ),  # Set colorbar range
    #             cmax=max(fitness_values) if fitness_values else 1,
    #             size=25,
    #             line_width=3,
    #         ),
    #         showlegend=True,
    #     )

    #     # Create node trace for non-highlighted nodes
    #     node_trace_other = go.Scatter(
    #         x=node_x_other,
    #         y=node_y_other,
    #         mode="markers+text",
    #         hoverinfo="text",
    #         text=node_labels_other,
    #         hovertext=node_text_other,
    #         textposition="middle center",
    #         marker=dict(
    #             showscale=False,
    #             color="grey",
    #             size=20,
    #             line_width=2,
    #             opacity=0.5,
    #         ),
    #         showlegend=False,
    #     )
    #     data = [
    #         edge_trace_other,
    #         edge_trace_highlighted,
    #         node_trace_other,
    #         node_trace_highlighted,
    #     ]
    else:
        # Create node trace for all nodes
        node_trace_other = go.Scatter(
            x=node_x_other,
            y=node_y_other,
            mode="markers+text",
            hoverinfo="text",
            text=node_labels_other,
            hovertext=node_text_other,
            textposition="middle center",
            marker=dict(
                showscale=False,
                color=node_colors,
                size=20,
                line_width=2,
            ),
            showlegend=False,
        )
        data = [edge_trace_other, node_trace_other]

    # Calculate optimal height based on number of generations
    # max_gen = max(nodes_by_generation.keys()) if nodes_by_generation else 0
    # optimal_height = max(400, 100 + max_gen * 50)

    # Create the figure with all traces
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title="Model Evolution Tree by Generation",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                title="Generation",
            ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


def get_continuous_coevolution_scaling_law_data(
    best_of_N_results_per_benchmark_per_selection_method: dict,
    averaged_across_benchmarks_results: dict,
) -> dict:
    """
    Get the continuous coevolution scaling law data per benchmark and averaged across benchmarks.

    Returns:
        dict[str, dict[str, dict[str, dict[str, float]]]]: A dictionary of dictionaries containing the continuous coevolution scaling law data per benchmark and averaged across benchmarks for different number of models N.
        In the format of:
        --- Per benchmark ---
        {
            "averaged_across_benchmarks": {
                <N>: {
                    <gen_num_1>: <best_of_N_accuracy>,
                    <gen_num_2>: <best_of_N_accuracy>,
                    ...
                },
                ...
            },
            <benchmark_name_1>: {
                <N>: {
                    <gen_num_1>: <best_of_N_accuracy>,
                    <gen_num_2>: <best_of_N_accuracy>,
                    ...
                },
                ...
            },
            <benchmark_name_2>: {
                <N>: {
                    <gen_num_1>: <best_of_N_accuracy>,
                    <gen_num_2>: <best_of_N_accuracy>,
                    ...
                },
                ...
            },
        }
    """

    # Get the data for each benchmark
    def get_data_for_benchmark(
        data_for_benchmark: dict,
    ) -> dict:
        # Store the results for all N for this benchmark
        results_per_N = {}

        for (
            selection_method,
            data_for_selection_method,
        ) in data_for_benchmark.items():
            # The selection methods have the same name structure:
            # gsvc_max_gen_<gen_num>_w_task_filtering
            # We want to extract the gen_num
            re_pattern = r"gsvc_max_gen_(\d+)_w_task_filtering"
            match = re.search(re_pattern, selection_method)
            if match:
                gen_num = int(match.group(1))

            # Get the results for each N
            for N, data_for_N in data_for_selection_method.items():
                if N not in results_per_N:
                    results_per_N[N] = {}
                results_per_N[N][gen_num] = data_for_N["best_of_N_accuracy"]

        return results_per_N

    # Get the data for each benchmark
    per_benchmark_data = {
        benchmark_name: get_data_for_benchmark(data_for_benchmark)
        for benchmark_name, data_for_benchmark in best_of_N_results_per_benchmark_per_selection_method.items()
    }

    # Get the data averaged across benchmarks
    per_benchmark_data["averaged_across_benchmarks"] = get_data_for_benchmark(
        averaged_across_benchmarks_results
    )

    return per_benchmark_data


def get_continuous_coevolution_scaling_law_plot(
    continuous_coevolution_scaling_law_data: dict,
    benchmark_name: str,
    do_log_scale: bool = False,
    selected_N: list[int] = None,
    polynomial_degree: int = 1,
) -> go.Figure:
    """Get the continuous coevolution scaling law plot.

    Args:
        continuous_coevolution_scaling_law_data: The continuous coevolution scaling law data.
            In the format of:
            {
                <N>: {
                    <gen_num_1>: <best_of_N_accuracy>,
                    <gen_num_2>: <best_of_N_accuracy>,
                    ...
                },
                ...
            }
        benchmark_name: The name of the benchmark.
        do_log_scale: Whether to use a log scale for the x-axis.
        selected_N: The N values to plot.

    Returns:
        go.Figure: The continuous coevolution scaling law plot.
            x-axis: Generation number
            y-axis: Accuracy
            title: Continuous coevolution scaling law plot: <benchmark_name>
            legend title: Number of models in task force
    """

    # Plot a line for each N
    fig = go.Figure()
    color_idx = 0
    for N, data in continuous_coevolution_scaling_law_data.items():
        if selected_N is not None and N not in selected_N:
            continue
        fig.add_trace(
            go.Scatter(
                x=list(data.keys()),
                y=list(data.values()),
                name=f"N={N}",
                marker=dict(
                    size=10,
                    color=COLOR_PALETTE[color_idx],
                ),
            )
        )

        # Fit linear regression to the data
        linear_regression_fit = np.polyfit(
            list(data.keys()), list(data.values()), polynomial_degree
        )
        linear_regression_line = np.polyval(
            linear_regression_fit, list(data.keys())
        )

        # Plot the linear regression line
        fig.add_trace(
            go.Scatter(
                x=list(data.keys()),
                y=linear_regression_line,
                name=f"Linear regression (N={N})",
                marker=dict(
                    size=10,
                    color=COLOR_PALETTE[color_idx],
                ),
                opacity=0.5,
                line=dict(
                    dash="dash",
                    width=2,
                ),
                showlegend=False,
            )
        )
        color_idx += 1

    fig.update_layout(
        title=f"Continuous coevolution scaling law plot: {benchmark_name}",
        xaxis_title=(
            "Generation number"
            if not do_log_scale
            else "log(Generation number)"
        ),
        yaxis_title="Coverage",
        legend_title="Number of models in task force",
    )

    if do_log_scale:
        fig.update_xaxes(type="log")

    return fig


def main():
    experiment_path = "outputs/2025-05-27/08-31-08"
    compute_and_load_synth_task_to_tSNE_mapping(experiment_path)


if __name__ == "__main__":
    main()
