import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import numpy as np
import logging
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compute_vendi_scores import compute_vendi_scores
from utils import (
    build_task_hover_text,
    get_task_tSNE_graph_object,
    compute_and_load_synth_task_to_tSNE_mapping,
    load_generation_files,
    get_adaptation_generations_and_gen_step_size,
    get_global_min_max_values,
    create_task_embedding_plot_for_generation,
    COLOR_PALETTE,
    create_task_embedding_gif,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the evolution history of a task family."
    )
    parser.add_argument(
        "--seed_task_dir",
        type=str,
        default=os.path.join(project_dir, "seed_tasks"),
        help="Seed task directory to visualize (e.g., 'math_word_problem_clips'). If not provided, will prompt.",
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to the directory containing generated task families.",
    )
    parser.add_argument(
        "--initial_task",
        type=str,
        default="task_263_identify_flooding_factors",
        help="Task number or name to visualize (e.g., 'task_5_addition'). If not provided, will prompt.",
    )
    return parser.parse_args()


def find_task_dirs(tasks_pool_dir: str, is_seed: bool = False) -> Dict[str, Dict]:
    """
    Returns a mapping from task dir name to its metadata and file paths.
    """
    task_dirs = {}
    for entry in os.listdir(tasks_pool_dir):
        full_path = os.path.join(tasks_pool_dir, entry)
        if os.path.isdir(full_path) and (entry.startswith("task_") or is_seed):
            meta_path = os.path.join(full_path, "metadata.json")
            task_json_path = os.path.join(full_path, "task.json")
            task_py_path = os.path.join(full_path, "task.py")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                task_dirs[entry] = {
                    "dir": full_path,
                    "metadata": metadata,
                    "task_json": (
                        task_json_path if os.path.exists(task_json_path) else None
                    ),
                    "task_py": (task_py_path if os.path.exists(task_py_path) else None),
                }
    return task_dirs


def count_adaptation_types(task_dirs: Dict[str, Dict]) -> Dict[str, int]:
    """
    Counts the number of adaptation types for each task.
    """
    count_adaptation_types = {}
    for task in task_dirs:
        adaptation_type = task_dirs[task]["metadata"].get(
            "generation_type", "seed_task"
        )
        if adaptation_type not in count_adaptation_types:
            count_adaptation_types[adaptation_type] = 0
        count_adaptation_types[adaptation_type] += 1

    return count_adaptation_types


def trace_evolution(task_dirs: Dict[str, Dict], start_task: str) -> List[str]:
    """
    Returns a list of task dir names representing the evolution chain, from root to the selected task.
    """
    chain = []
    pass_rate_mapping = {}
    current = start_task
    while current:
        chain.append(current)
        meta = task_dirs[current]["metadata"]
        prev = meta.get("generated_from_task")
        if prev and prev in task_dirs:
            current = prev
            pass_rate_mapping[prev] = meta.get("original_pass_rate", "N/A")
        # check if the task was generated from a seed task
        elif meta.get("generated_from_seed", None):
            prev = meta["generated_from_seed"]
            current = prev
            pass_rate_mapping[prev] = meta.get("original_pass_rate", "N/A")

            # the new batched initial task generation screws up the "generated_from_seed" field.
            # It's not the original seed task, it's a task from a previous batch.
            if prev not in task_dirs:
                # this means it is the true seed task, which is not in the task_dirs
                # add this as the final task in the chain and break
                chain.append(prev)
                break

        else:
            break
    return list(reversed(chain)), pass_rate_mapping


def load_task_data(task_dirs: Dict[str, Dict], chain: List[str]) -> List[Dict]:
    """
    Loads metadata, description, and code for each task in the chain.
    """
    data = []
    for task in chain:
        if task not in task_dirs:
            data.append(
                {
                    "name": task,
                    "metadata": {},
                    "description": {},
                    "code": "",
                }
            )
            continue
        entry = task_dirs[task]
        with open(entry["task_json"], "r") as f:
            description = json.load(f)
        with open(entry["task_py"], "r") as f:
            code = f.read()
        data.append(
            {
                "name": task,
                "metadata": entry["metadata"],
                "description": description,
                "code": code,
            }
        )
    return data


def build_pass_rate_mapping(task_dirs: Dict[str, Dict]) -> Dict[str, float]:
    """
    Builds a mapping from task name to its actual pass rate by looking at
    what other tasks say about it in their 'original_pass_rate' field.
    """
    pass_rate_mapping = {}

    for task_name, task_data in task_dirs.items():
        metadata = task_data["metadata"]

        # If this task was generated from another task, record that task's pass rate
        if "generated_from_task" in metadata and "original_pass_rate" in metadata:
            # DEBUGGING
            if (
                "task_95_explain_flooding_cause"
                # "task_263_identify_flooding_factors"
                in metadata["generated_from_task"]
            ):
                print(metadata)

            parent_task = metadata["generated_from_task"]
            pass_rate = metadata["original_pass_rate"]
            if not parent_task in pass_rate_mapping:
                pass_rate_mapping[parent_task] = []
            pass_rate_mapping[parent_task].append(pass_rate)

    return pass_rate_mapping


def show_flow_chart(
    data: List[Dict],
    pass_rate_mapping: Dict[str, float] = None,
):
    """
    Visualizes the evolution chain as a flow chart with interactive details.
    Now uses the pass_rate_mapping to show each task's actual pass rate.
    """
    st.header("Task Evolution History")
    G = nx.DiGraph()
    labels = {}
    for i, node in enumerate(data):
        name = node["name"]
        meta = node["metadata"]
        if name.startswith("task_"):
            # Use the actual pass rate for this task from the mapping
            actual_pass_rate = (
                pass_rate_mapping.get(name, "N/A")
                if pass_rate_mapping
                else meta.get("original_pass_rate", "N/A")
            )
            label = f"{name}\nType: {meta['generation_type']}\nPass Rate: {actual_pass_rate}"
        else:
            # For seed tasks, we might not have pass rates
            actual_pass_rate = (
                pass_rate_mapping.get(name, "N/A") if pass_rate_mapping else "N/A"
            )
            label = f"{name}\nType: Seed Task\nPass Rate: {actual_pass_rate}"
        labels[name] = label
        if i > 0:
            G.add_edge(data[i - 1]["name"], name)
        else:
            G.add_node(name)

    # Replace spring_layout with custom left-to-right positioning with zig-zag pattern
    pos = {}
    for i, node in enumerate(data):
        name = node["name"]
        # Create multi-row zig-zag pattern
        row = i % 2  # 0 for top rows, 1 for bottom rows

        # Increase horizontal spacing between nodes
        x_pos = i * 5

        if row == 0:  # Top rows
            # Alternate between y=20 and y=13 for top rows
            sub_row = (i // 2) % 2  # 0 for y=20, 1 for y=13
            y_pos = 20 if sub_row == 0 else 13
        else:  # Bottom rows
            # Alternate between y=-20 and y=-13 for bottom rows
            sub_row = (i // 2) % 2  # 0 for y=-20, 1 for y=-13
            y_pos = -20 if sub_row == 0 else -13

        pos[name] = (x_pos, y_pos)

    # # Increase k parameter to push nodes further apart
    # pos = nx.spring_layout(G, k=2.0, iterations=50)
    fig, ax = plt.subplots(figsize=(18, 9))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=3000,
        node_color="lightblue",
        ax=ax,
    )
    # Increase margins to prevent text cutoff
    plt.margins(x=0.2, y=0.2)
    st.pyplot(fig)

    st.header("Task Details")
    for node in data:
        st.subheader(node["name"])
        st.write("**Example Instruction:**")
        instruction_text = node["description"]["example_instruction"]
        # replace $ signs with \$
        instruction_text = instruction_text.replace("$", "\\$")
        st.write(instruction_text)
        meta_w_o_embedding = node["metadata"].copy()
        meta_w_o_embedding.pop("embedding", None)
        st.json(meta_w_o_embedding)
        st.write("**Description:**")
        st.json(node["description"])
        st.write("**Code:**")
        st.code(node["code"], language="python")


def get_new_tasks_per_generation(
    generations: Dict[int, List[str]],
) -> Dict[int, List[str]]:
    """
    For each generation, returns only the new tasks that weren't in the previous generation.
    """
    new_tasks = {}
    tasks_discarded_at_generation = {}
    all_past_tasks = set()

    for gen_num, tasks in generations.items():
        current_tasks = set(tasks)

        ### All new tasks that were added at this generation
        new_tasks_at_generation = set()
        for task in current_tasks:
            if task not in all_past_tasks:
                new_tasks_at_generation.add(task)
        new_tasks[gen_num] = new_tasks_at_generation
        all_past_tasks.update(new_tasks_at_generation)

        ### All tasks that were discarded at this generation
        missing_tasks_at_generation = set()
        for task in all_past_tasks:
            if task not in current_tasks:
                missing_tasks_at_generation.add(task)
        tasks_discarded_at_generation[gen_num] = missing_tasks_at_generation

    # convert sets to lists
    for gen_num, tasks in new_tasks.items():
        new_tasks[gen_num] = list(tasks)
    for gen_num, tasks in tasks_discarded_at_generation.items():
        tasks_discarded_at_generation[gen_num] = list(tasks)

    return new_tasks


def create_generation_tree(
    task_dirs: Dict[str, Dict],
    tasks_pool_dir: str,
    seed_task_dir: str,
) -> nx.DiGraph:
    """
    Creates a left-to-right tree based on generation files.
    """
    G = nx.DiGraph()

    # Load generation files and get new tasks per generation
    generations = load_generation_files(tasks_pool_dir)
    new_tasks_per_gen = get_new_tasks_per_generation(generations)

    # Add seed tasks to generation 0
    for task in task_dirs:
        if not task.startswith("task_"):
            # if 0 not in new_tasks_per_gen:
            #     new_tasks_per_gen[0] = []
            # new_tasks_per_gen[0].append(seed_task_dir + "/" + task)
            if -1 not in new_tasks_per_gen:
                new_tasks_per_gen[-1] = []
            new_tasks_per_gen[-1].append(seed_task_dir + "/" + task)

    # sort the dictionary by generation number
    new_tasks_per_gen = dict(sorted(new_tasks_per_gen.items()))
    all_tasks_in_generations = set()
    for gen_num, tasks in new_tasks_per_gen.items():
        for task in tasks:
            all_tasks_in_generations.add(task.split("/")[-1])

    # Add nodes and edges for each generation
    for gen_num, new_tasks in new_tasks_per_gen.items():
        if not new_tasks:  # Skip if no new tasks in this generation
            continue

        # Add nodes for this generation
        for task in new_tasks:
            task_name = task.split("/")[-1]
            if task_name in task_dirs:
                G.add_node(task_name, generation=gen_num, **task_dirs[task_name])

        # Add edges from previous generation
        # if gen_num > 0:
        if gen_num >= 0:
            prev_gen = gen_num - 1
            while prev_gen >= 0 and not new_tasks_per_gen.get(prev_gen):
                prev_gen -= 1

            if prev_gen >= -1:
                for prev_task in new_tasks_per_gen[prev_gen]:
                    prev_task_name = prev_task.split("/")[-1]
                    for task_name in task_dirs:
                        if task_name not in all_tasks_in_generations:
                            continue
                        meta = task_dirs[task_name]["metadata"]
                        if (
                            meta.get("generated_from_task") == prev_task_name
                            or meta.get("generated_from_seed") == prev_task_name
                        ):
                            G.add_edge(prev_task_name, task_name)

    return G


def calculate_optimal_height(G: nx.DiGraph, min_node_spacing: float = 1.5) -> int:
    """
    Calculate the optimal height for the visualization to prevent node overlap.

    Args:
        G: The networkx graph containing the nodes
        min_node_spacing: Minimum spacing between nodes as a multiplier of node size (default: 1.5)

    Returns:
        int: The calculated optimal height in pixels
    """
    # Get number of nodes per generation
    nodes_by_generation = {}
    for node in G.nodes():
        gen = G.nodes[node].get("generation", 0)
        if gen not in nodes_by_generation:
            nodes_by_generation[gen] = []
        nodes_by_generation[gen].append(node)

    # Find the generation with the most nodes
    max_nodes_in_generation = max(len(nodes) for nodes in nodes_by_generation.values())

    # Calculate base height needed for nodes
    # Each node is 10px in diameter, and we want at least min_node_spacing * node_size between nodes
    node_size = 10  # Size of nodes in pixels
    min_spacing = node_size * min_node_spacing

    # Calculate total height needed for nodes and spacing
    total_node_height = max_nodes_in_generation * (node_size + min_spacing)

    # Add padding for margins and title
    top_margin = 60  # Space for title
    bottom_margin = 40  # Space for x-axis labels
    vertical_padding = 100  # Additional padding for visual comfort

    # Calculate final height
    optimal_height = int(
        total_node_height + top_margin + bottom_margin + vertical_padding
    )

    # Ensure minimum height
    min_height = 800
    return max(optimal_height, min_height)


def create_interactive_generation_tree(
    G: nx.DiGraph,
    task_dirs: Dict[str, Dict],
    highlight_chain: Optional[List[str]] = None,
    positions: Dict[str, Tuple[float, float]] = None,
    pass_rate_mapping_all: Dict[str, float] = None,
    min_node_spacing: float = 0.5,
    absolute_height: int = None,
):
    """
    Creates an interactive tree visualization with left-to-right layout.
    Colors nodes based on their seed ancestor and turns non-highlighted nodes grey.
    """
    # Use hierarchical layout with left-to-right orientation
    pos = positions or nx.spring_layout(G, k=2, iterations=50)

    ### Adjust x-coordinates based on generation
    for node in G.nodes():
        gen = G.nodes[node].get("generation", 0)
        pos[node] = (gen, pos[node][1])  # Use generation number as x-coordinate

    ### For the nodes in each generation, space them out evenly
    # Get the nodes in each generation
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
        spacing = spacing_by_generation[gen]
        for i, node in enumerate(nodes):
            pos[node] = (pos[node][0], min_y + i * spacing)

    ### Create edge traces for highlighted and non-highlighted edges
    edge_x_highlighted = []
    edge_y_highlighted = []
    edge_x_other = []
    edge_y_other = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if (
            highlight_chain
            and edge[0] in highlight_chain
            and edge[1] in highlight_chain
        ):
            edge_x_highlighted.extend([x0, x1, None])
            edge_y_highlighted.extend([y0, y1, None])
        else:
            edge_x_other.extend([x0, x1, None])
            edge_y_other.extend([y0, y1, None])

    # Create edge traces with different colors
    edge_trace_highlighted = go.Scatter(
        x=edge_x_highlighted,
        y=edge_y_highlighted,
        # line=dict(width=1.5, color="#888"),
        line=dict(width=1.5, color="red"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    edge_trace_other = go.Scatter(
        x=edge_x_other,
        y=edge_y_other,
        line=dict(width=0.8, color="#ccc"),  # Lighter grey for non-highlighted edges
        opacity=0.5,
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
    # Determine seed ancestor for each node and assign colors
    all_seed_ancestors = set()
    node_to_seed_ancestor = {}

    for node_name in G.nodes():
        chain_to_root, _ = trace_evolution(task_dirs, node_name)
        if chain_to_root:
            seed_ancestor = chain_to_root[0]
        else:
            seed_ancestor = node_name
        all_seed_ancestors.add(seed_ancestor)
        node_to_seed_ancestor[node_name] = seed_ancestor

    seed_color_map = {
        seed: COLOR_PALETTE[i % len(COLOR_PALETTE)]
        for i, seed in enumerate(sorted(list(all_seed_ancestors)))
    }

    for node in G.nodes():
        x, y = pos[node]
        node_data = G.nodes[node]
        meta = node_data.get("metadata", {})
        task_json = node_data.get("task_json", None)
        if task_json:
            with open(task_json, "r") as f:
                description = json.load(f)

            example_instruction = description.get("example_instruction", "N/A")
            ## Format the example to have a max of 100 characters per new line
            new_example_instruction = ""
            all_words = example_instruction.split(" ")
            words_per_line = 15
            for i in range(0, len(all_words), words_per_line):
                end = min(i + words_per_line, len(all_words))
                new_example_instruction += " ".join(all_words[i:end]) + "<br>"
            example_instruction = new_example_instruction
        else:
            example_instruction = "N/A"

        # Create hover text with full details
        if node.startswith("task_"):
            hover_text = f"Task: {node}<br>Generation: {node_data.get('generation', 'N/A')}<br>Type: {meta.get('generation_type', 'N/A')}<br>Pass Rates: {pass_rate_mapping_all.get(node, 'N/A')}"
            # hover_text = f"Task: {node}<br>Generation: {node_data.get('generation', 'N/A')}<br>Type: {meta.get('generation_type', 'N/A')}<br>Pass Rate: {pass_rate_mapping.get(node, 'N/A')}<br>Example Instruction: {example_instruction}"
            label = node.split("_")[1]
        else:
            hover_text = f"Seed Task: {node}<br>Generation: 0"
            label = node

        if highlight_chain and node in highlight_chain:
            node_x_highlighted.append(x)
            node_y_highlighted.append(y)
            node_text_highlighted.append(hover_text)
            node_labels_highlighted.append(label)
            node_colors.append("red")
        else:
            node_x_other.append(x)
            node_y_other.append(y)
            node_text_other.append(hover_text)
            node_labels_other.append(label)
            node_colors.append(seed_color_map[node_to_seed_ancestor[node]])

    # Create node traces with different colors if a specific task is selected
    if highlight_chain:
        node_trace_highlighted = go.Scatter(
            x=node_x_highlighted,
            y=node_y_highlighted,
            mode="markers+text",
            hoverinfo="text",
            text=node_labels_highlighted,
            hovertext=node_text_highlighted,
            textposition="middle center",
            marker=dict(
                showscale=False,
                color="red",  # Highlighted nodes are red
                size=20,
                line_width=2,
            ),
            showlegend=False,
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
                color="grey",  # Non-highlighted nodes are grey
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
    else:
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
                color=node_colors,
                size=20,
                line_width=2,
            ),
            showlegend=False,
        )
        data = [edge_trace_other, node_trace_other]

    # Calculate optimal height
    if not absolute_height:
        figure_height = calculate_optimal_height(G, min_node_spacing=min_node_spacing)
    else:
        figure_height = absolute_height

    # Create the figure with all traces
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title="Task Evolution Tree by Generation",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            height=figure_height,
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


def show_adaptation_types_pie_chart(task_dirs: Dict[str, Dict]):
    num_adaptation_types = count_adaptation_types(task_dirs)
    df = pd.DataFrame([num_adaptation_types])
    # reporder the columns to be in the order of the adaptation types
    # seed tasks, initial, adaptation_easier, adaptation_harder, adaptation_novel
    adaptation_types = [
        "seed_task",
        "initial",
        "adaptation_easier",
        "adaptation_harder",
        "adaptation_novel",
    ]
    existing_adaptation_types = [col for col in adaptation_types if col in df.columns]

    df = df[existing_adaptation_types]

    # Define colors for each adaptation type
    color_map = {
        "seed_task": "#0068c9",  # Blue
        "initial": "#ffabab",  # Peach
        "adaptation_easier": "#7defa1",  # light green
        "adaptation_harder": "#ff2b2b",  # Red
        "adaptation_novel": "#83c9ff",  # Light Blue
    }

    fig = px.pie(
        df,
        values=list(df.values[0]),
        names=list(df.columns),
        hover_name=list(df.columns),  # Show category names on hover
        color=list(df.columns),
        color_discrete_map=color_map,
    )
    st.plotly_chart(fig)


def count_adaptation_types_per_generation(
    task_dirs: Dict[str, Dict], G: nx.DiGraph
) -> pd.DataFrame:
    """
    Counts the number of each adaptation type per generation.
    Returns a DataFrame with columns: generation, adaptation_type, count
    """
    # Initialize dictionary to store counts per generation
    gen_counts = {}

    # Get all generations
    generations = sorted(set(G.nodes[node].get("generation", 0) for node in G.nodes()))

    # Initialize counts for each generation
    for gen in generations:
        gen_counts[gen] = {
            "adaptation_easier": 0,
            "adaptation_harder": 0,
            "adaptation_novel": 0,
        }

    # Count adaptation types for each node in each generation
    for node in G.nodes():
        gen = G.nodes[node].get("generation", 0)
        meta = task_dirs[node]["metadata"]
        adapt_type = meta.get("generation_type", "seed_task")
        # Only count adaptation types, skip seed_task and initial
        if adapt_type in [
            "adaptation_easier",
            "adaptation_harder",
            "adaptation_novel",
        ]:
            gen_counts[gen][adapt_type] += 1

    # Convert to DataFrame
    data = []
    for gen in generations:
        for adapt_type, count in gen_counts[gen].items():
            data.append(
                {
                    "generation": gen,
                    "adaptation_type": adapt_type,
                    "count": count,
                }
            )

    return pd.DataFrame(data)


def load_vendi_scores(tasks_pool_dir: str) -> Dict[int, float]:
    """
    Loads Vendi scores from vendi_scores_per_generation.json in the tasks_pool_dir.
    Returns a dictionary mapping generation numbers to scores.
    """
    local_vendi_scores_path = os.path.join(
        tasks_pool_dir, "vendi_scores_per_generation.json"
    )
    global_vendi_scores_path = os.path.join(
        tasks_pool_dir, "global_vendi_scores_per_generation.json"
    )

    vendi_score = None
    if not os.path.exists(local_vendi_scores_path) or not os.path.exists(
        global_vendi_scores_path
    ):
        st.info(
            f"Local or global Vendi scores file not found at\n{local_vendi_scores_path}\nor\n{global_vendi_scores_path}. Computing Vendi scores..."
        )
        local_vendi_scores, global_vendi_scores = compute_vendi_scores(tasks_pool_dir)
        vendi_score = {
            "local": local_vendi_scores,
            "global": global_vendi_scores,
        }

    if vendi_score:
        return vendi_score

    try:
        with open(local_vendi_scores_path, "r") as f:
            local_vendi_scores = {int(k): float(v) for k, v in json.load(f).items()}
        with open(global_vendi_scores_path, "r") as f:
            global_vendi_scores = {int(k): float(v) for k, v in json.load(f).items()}

        return {
            "local": local_vendi_scores,
            "global": global_vendi_scores,
        }
    except FileNotFoundError:
        st.warning(
            f"Vendi scores file not found at {local_vendi_scores_path} or {global_vendi_scores_path}"
        )
        return {}
    except json.JSONDecodeError:
        st.warning(
            f"Invalid JSON format in {local_vendi_scores_path} or {global_vendi_scores_path}"
        )
        return {}


def count_active_tasks_per_generation(tasks_pool_dir: str) -> Dict[int, int]:
    """
    Counts the number of active tasks per generation from active_pool_gen_<gen>.json files.
    Returns a dictionary mapping generation numbers to counts.
    """
    active_tasks = {}
    global_active_tasks = {}
    all_tasks = set()

    relevant_files = glob.glob(os.path.join(tasks_pool_dir, "active_pool_gen_*.json"))
    relevant_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    for file in relevant_files:
        gen_num = int(file.split("_")[-1].split(".")[0])
        with open(file, "r") as f:
            tasks = json.load(f)
            active_tasks[gen_num] = len(tasks)
            all_tasks.update([task.split("/")[-1] for task in tasks])
            global_active_tasks[gen_num] = len(all_tasks)
    return active_tasks, global_active_tasks


def show_adaptation_types_line_graph(
    task_dirs: Dict[str, Dict], G: nx.DiGraph, tasks_pool_dir: str
):
    """
    Shows a line graph of adaptation types over generations, with optional Vendi scores on a second y-axis.
    Shows a separate plot for active tasks percentage below.
    """
    df = count_adaptation_types_per_generation(task_dirs, G)

    # Define colors for each adaptation type (only for adaptation types)
    color_map = {
        "adaptation_easier": "#7defa1",  # light green
        "adaptation_harder": "#ff2b2b",  # Red
        "adaptation_novel": "#83c9ff",  # Light Blue
    }

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add adaptation type lines
    for adapt_type in df["adaptation_type"].unique():
        df_type = df[df["adaptation_type"] == adapt_type]
        fig.add_trace(
            go.Scatter(
                x=df_type["generation"],
                y=df_type["count"],
                name=adapt_type,
                line=dict(color=color_map[adapt_type]),
            ),
            secondary_y=False,
        )

    # Add Vendi scores if available
    vendi_scores = None
    max_gen = max(df["generation"].unique())
    # try:
    vendi_scores = load_vendi_scores(tasks_pool_dir)
    if vendi_scores:

        # Check box to show local or global Vendi scores
        show_local_vendi_scores = st.checkbox(
            "Show Local Vendi Scores", value=True, key="show_local_vendi_scores"
        )
        show_global_vendi_scores = st.checkbox(
            "Show Global Vendi Scores",
            value=True,
            key="show_global_vendi_scores",
        )

        # Add Vendi scores for each type
        dash_type = {
            "local": "dash",
            "global": "dot",
        }
        for key, scores in vendi_scores.items():
            if not show_local_vendi_scores and key == "local":
                continue
            if not show_global_vendi_scores and key == "global":
                continue
            generations = sorted(scores.keys())
            scores = [scores[gen] for gen in generations if gen <= max_gen]
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=scores,
                    name=f"{key} Vendi Score",
                    line=dict(
                        color="#9467bd", dash=dash_type[key]
                    ),  # Purple dashed line
                ),
                secondary_y=True,
            )

    # except Exception as e:
    #     st.warning(f"Error loading Vendi scores: {e}")

    # Update layout for main plot
    fig.update_layout(
        title="Evolution of Adaptation Types and Vendi Scores Over Generations",
        xaxis=dict(
            title="Generation",
            tickmode="array",
            tickvals=sorted(set(df["generation"].unique())),
            ticktext=[str(gen) for gen in sorted(set(df["generation"].unique()))],
        ),
        showlegend=True,
    )

    # Update y-axes labels and remove grid lines
    fig.update_yaxes(
        title_text="Number of New Tasks",
        secondary_y=False,
        showgrid=True,  # Remove horizontal grid lines
    )
    if vendi_scores:
        fig.update_yaxes(
            title_text="Vendi Score",
            secondary_y=True,
            showgrid=False,  # Remove horizontal grid lines
        )

    st.plotly_chart(fig)

    # Create separate plot for active tasks
    active_tasks, global_active_tasks = count_active_tasks_per_generation(
        tasks_pool_dir
    )
    if active_tasks:
        fig_active = go.Figure()
        generations = sorted(active_tasks.keys())

        # Local active tasks
        counts = [active_tasks[gen] for gen in generations]

        # Add active tasks line
        fig_active.add_trace(
            go.Scatter(
                x=generations,
                y=counts,
                name="Local Active Tasks",
                line=dict(color="#ff7f0e", width=2),  # Orange line
            )
        )

        # Update layout for active tasks plot
        fig_active.update_layout(
            title="Active Tasks Over Generations",
            xaxis=dict(
                title="Generation",
                tickmode="array",
                tickvals=sorted(set(df["generation"].unique())),
                ticktext=[str(gen) for gen in sorted(set(df["generation"].unique()))],
            ),
            yaxis=dict(
                title="Local Active Tasks",
            ),
            showlegend=True,
        )

        # Global active tasks
        counts = [global_active_tasks[gen] for gen in generations]
        fig_active.add_trace(
            go.Scatter(
                x=generations,
                y=counts,
                name="Task Archive",
                line=dict(color="#1f77b4", width=2),  # Blue line
            ),
        )
        st.plotly_chart(fig_active)


def task_embedding_evolution_gif_button(
    experiment_path: str,
):
    # Add GIF creation button
    st.write("Create Task Embedding Evolution GIF")
    col1, col2, col3 = st.columns(3)

    if "gif_path" not in st.session_state:
        st.session_state.gif_path = None

    with col1:
        fps = st.slider(
            "FPS",
            min_value=1,
            max_value=5,
            value=2,
            help="Frames per second for the GIF",
        )
        include_all = st.checkbox(
            "Include 'All Generations' frame",
            value=True,
            help="Add a final frame showing all generations together",
        )

    with col2:
        if st.button("Generate GIF", type="primary"):
            with st.spinner("Creating task embedding evolution GIF..."):
                try:
                    st.session_state.gif_path = create_task_embedding_gif(
                        experiment_path=experiment_path,
                        fps=fps,
                        include_all_generation=include_all,
                    )

                    if st.session_state.gif_path and os.path.exists(
                        st.session_state.gif_path
                    ):
                        st.success(f"GIF created successfully!")
                    else:
                        st.error(
                            "Failed to create GIF. Please check the logs for more information."
                        )

                except Exception as e:
                    st.error(f"Error creating GIF: {str(e)}")
                    logger.error(f"Error creating GIF: {e}")

        if st.session_state.gif_path and os.path.exists(st.session_state.gif_path):
            # download the gif button
            with open(st.session_state.gif_path, "rb") as file:
                gif_bytes = file.read()

            st.download_button(
                label="Download GIF",
                data=gif_bytes,
                file_name="task_embedding_evolution.gif",
                mime="image/gif",
            )

    with col3:
        # show the gif in a loop if button was clicked
        if st.button("Show GIF"):
            if "gif_path" not in st.session_state:
                st.error("No GIF created yet. Please create a GIF first.")
                return

            # get the gif path from the session state
            gif_path = st.session_state.gif_path

            # Show the GIF in the app
            st.image(
                gif_path,
                caption="Task Embedding Evolution GIF",
                use_container_width=True,
            )


def show_task_embedding_evolution_per_gen(
    task_name_to_tSNE_embedding: Dict[str, np.ndarray],
    task_name_to_hdbscan_cluster: Dict[str, int],
    experiment_path: str,
    relevant_gen: int,
    all_tasks_graph: nx.Graph = None,
    x_axis_range: Tuple[float, float] = None,
    y_axis_range: Tuple[float, float] = None,
):
    """
    Shows the evolution of task embeddings per generation.
    """
    fig, title = create_task_embedding_plot_for_generation(
        task_name_to_tSNE_embedding=task_name_to_tSNE_embedding,
        task_name_to_hdbscan_cluster=task_name_to_hdbscan_cluster,
        experiment_path=experiment_path,
        relevant_gen=relevant_gen,
        all_tasks_graph=all_tasks_graph,
        x_axis_range=x_axis_range,
        y_axis_range=y_axis_range,
    )

    st.subheader(title)
    st.plotly_chart(fig)
    task_embedding_evolution_gif_button(experiment_path)


def show_evolution_tree(
    task_dirs: Dict[str, Dict],
    G: nx.DiGraph,
):
    """
    Shows the complete evolution tree with interactive features.
    """
    st.title("Task Evolution Tree by Generation")
    # Add task selection dropdown first with "All" option
    task_options = ["All"] + list(task_dirs.keys())
    selected_task = st.selectbox(
        "Select a task to highlight its evolution chain", task_options
    )

    # Slider to select the minimum node spacing
    min_node_spacing = st.slider(
        "Minimum Vertical Node Spacing",
        min_value=0.01,
        max_value=1.0,
        value=0.5,
        help="The minimum vertical spacing between nodes in the tree. "
        "This is used to control the height of the tree. Lower values make the tree more compact vertically.",
    )

    # Slider to select the absolute height of the tree
    set_absolute_height = st.checkbox(
        "Select Absolute Height (if selected, min node spacing will be ignored)",
        value=True,
        key="set_absolute_height",
    )
    if set_absolute_height:
        absolute_height = st.slider(
            "Absolute Height",
            key="absolute_height",
            min_value=1000,
            max_value=5000,
            value=1000,
            step=100,
            help="The absolute height of the tree in pixels. If selected, the min node spacing will be ignored. "
            "This is used to control the height of the tree. Lower values make the tree more compact vertically.",
        )
    else:
        absolute_height = None

    # Calculate positions once and store them in the session state
    if "tree_positions" not in st.session_state:
        # Increase k and iterations for more spread out layout
        st.session_state.tree_positions = nx.spring_layout(
            G,
            k=5,
            iterations=100,  # Increased k from 2 to 5, iterations from 50 to 100
        )
        # Adjust x-coordinates based on generation
        for node in G.nodes():
            gen = G.nodes[node].get("generation", 0)
            st.session_state.tree_positions[node] = (
                gen,
                st.session_state.tree_positions[node][1],
            )

    # Get the chain to highlight if a task is selected (and it's not "All")
    highlight_chain = None
    pass_rate_mapping = None
    # debugging
    # selected_task = "task_1993_calculate_compound_interest"
    if selected_task and selected_task != "All":
        highlight_chain, pass_rate_mapping = trace_evolution(task_dirs, selected_task)

    # Create and display the interactive tree using stored positions
    pass_rate_mapping_all = build_pass_rate_mapping(task_dirs)
    fig = create_interactive_generation_tree(
        G,
        task_dirs,
        highlight_chain,
        st.session_state.tree_positions,
        pass_rate_mapping_all,
        min_node_spacing=min_node_spacing,
        absolute_height=absolute_height,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show the evolution chain and details for the selected task (if not "All")
    if selected_task and selected_task != "All":
        st.header(f"Evolution Chain for {selected_task}")
        chain, pass_rate_mapping = trace_evolution(task_dirs, selected_task)
        data = load_task_data(task_dirs, chain)
        show_flow_chart(
            data=data,
            pass_rate_mapping=pass_rate_mapping,
        )


def show_flow_chart_cli(data: List[Dict], pass_rate_mapping: Dict[str, float] = None):
    """
    Visualizes the evolution chain as a flow chart for CLI mode.
    Now uses the pass_rate_mapping to show each task's actual pass rate.
    """
    print("\nTask Evolution History")
    G = nx.DiGraph()
    labels = {}
    for i, node in enumerate(data):
        name = node["name"]
        meta = node["metadata"]
        if name.startswith("task_"):
            # Use the actual pass rate for this task from the mapping
            actual_pass_rate = (
                pass_rate_mapping.get(name, "N/A")
                if pass_rate_mapping
                else meta.get("original_pass_rate", "N/A")
            )
            label = f"{name}\nType: {meta['generation_type']}\nPass Rate: {actual_pass_rate}"
        else:
            # For seed tasks, we might not have pass rates
            actual_pass_rate = (
                pass_rate_mapping.get(name, "N/A") if pass_rate_mapping else "N/A"
            )
            label = f"{name}\nType: Seed Task\nPass Rate: {actual_pass_rate}"
        labels[name] = label
        if i > 0:
            G.add_edge(data[i - 1]["name"], name)
        else:
            G.add_node(name)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=3000,
        node_color="lightblue",
    )
    plt.margins(x=0.2, y=0.2)
    plt.title("Task Evolution Chain")
    plt.show()

    print("\nTask Details:")
    for node in data:
        print(f"\n{node['name']}")
        print("Metadata:", json.dumps(node["metadata"], indent=2))
        print("Description:", json.dumps(node["description"], indent=2))
        print("Code:")
        print(node["code"])


def show_evolution_tree_cli(
    task_dirs: Dict[str, Dict],
    tasks_pool_dir: str,
    seed_task_dir: str,
    selected_task: Optional[str] = None,
):
    """
    Shows the complete evolution tree in CLI mode.
    """
    print("\nTask Evolution Tree by Generation")
    num_adaptation_types = count_adaptation_types(task_dirs)
    print(num_adaptation_types)
    G = create_generation_tree(
        task_dirs=task_dirs,
        tasks_pool_dir=tasks_pool_dir,
        seed_task_dir=seed_task_dir,
    )

    # Get the chain to highlight if a task is selected
    highlight_chain = None
    if selected_task:
        highlight_chain, actual_node_pass_rates = trace_evolution(
            task_dirs, selected_task
        )

    # Create visualization
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Adjust x-coordinates based on generation
    for node in G.nodes():
        gen = G.nodes[node].get("generation", 0)
        pos[node] = (gen, pos[node][1])  # Use generation number as x-coordinate

    plt.figure(figsize=(15, 8))

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)

    # Draw nodes with different colors for highlighted chain and seed ancestry
    # Determine seed ancestor for each node and assign colors
    all_seed_ancestors_cli = set()
    node_to_seed_ancestor_cli = {}

    for node_name in G.nodes():
        chain_to_root_cli, actual_node_pass_rates = trace_evolution(
            task_dirs, node_name
        )
        if chain_to_root_cli:
            seed_ancestor_cli = chain_to_root_cli[0]
        else:
            seed_ancestor_cli = node_name  # Fallback
        all_seed_ancestors_cli.add(seed_ancestor_cli)
        node_to_seed_ancestor_cli[node_name] = seed_ancestor_cli

    seed_color_map_cli = {
        seed: COLOR_PALETTE[i % len(COLOR_PALETTE)]
        for i, seed in enumerate(sorted(list(all_seed_ancestors_cli)))
    }

    node_colors_cli = []
    for node in G.nodes():
        base_color = seed_color_map_cli.get(
            node_to_seed_ancestor_cli.get(node, node), "grey"
        )
        if highlight_chain and node in highlight_chain:
            node_colors_cli.append("red")
        else:
            node_colors_cli.append(base_color)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors_cli, node_size=2000)

    # Draw labels - show full names for seed tasks, only IDs for generated tasks
    labels = {
        node: node.split("_")[-1] if node.startswith("task_") else node
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels)

    # Add margins to prevent text cutoff
    plt.margins(x=0.2, y=0.2)

    plt.title("Complete Task Evolution Tree by Generation")
    plt.xlabel("Generation")
    plt.axis("off")
    plt.show()


# ttl (seconds): here 60 * 60 = 1 hour
@st.cache_data(ttl=60 * 60)
def load_experiment_data(
    experiment_path: str,
    task_dirs: Dict[str, Dict],
    tasks_pool_dir: str,
    seed_task_dir: str,
) -> tuple[dict, dict, nx.DiGraph, dict, tuple[float, float], tuple[float, float]]:
    """Load experiment data with caching to avoid reloading on every rerun."""
    # tSNE embedding and HDBSCAN clusters
    try:
        task_name_to_tSNE_embedding, task_name_to_hdbscan_cluster = (
            compute_and_load_synth_task_to_tSNE_mapping(experiment_path)
        )
        # Create the graph object once with all embeddings
        G_EMBEDDINGS = get_task_tSNE_graph_object(task_name_to_tSNE_embedding)

        global_x_min, global_x_max, global_y_min, global_y_max = (
            get_global_min_max_values(task_name_to_tSNE_embedding, G_EMBEDDINGS)
        )
        x_axis_range = (global_x_min, global_x_max)
        y_axis_range = (global_y_min, global_y_max)

    except Exception as e:
        st.error(f"Error loading tSNE embedding and HDBSCAN clusters: {e}")
        task_name_to_tSNE_embedding = {}
        task_name_to_hdbscan_cluster = {}
        G_EMBEDDINGS = None
        x_axis_range = None
        y_axis_range = None

    # Generation tree
    try:
        G = create_generation_tree(
            task_dirs=task_dirs,
            tasks_pool_dir=tasks_pool_dir,
            seed_task_dir=seed_task_dir,
        )
    except Exception as e:
        st.error(f"Error loading generation tree: {e}")
        G = None
    return (
        task_name_to_tSNE_embedding,
        task_name_to_hdbscan_cluster,
        G,
        G_EMBEDDINGS,
        x_axis_range,
        y_axis_range,
    )


def main(args: argparse.Namespace):
    experiment_path = args.experiment_path
    tasks_pool_dir = os.path.join(experiment_path, "generated_tasks", "pool")

    experiment_name = "/".join(experiment_path.split("/")[-2:])

    # Get the task dirs from the tasks_pool_dir
    task_dirs = find_task_dirs(tasks_pool_dir)

    # Add seed task to task_dirs
    seed_task_dir = args.seed_task_dir
    seed_task_dirs = find_task_dirs(seed_task_dir, is_seed=True)
    for task in seed_task_dirs:
        task_dirs[task] = seed_task_dirs[task]

    # Load the tSNE embedding for the tasks
    (
        task_name_to_tSNE_embedding,
        task_name_to_hdbscan_cluster,
        G,
        G_EMBEDDINGS,
        x_axis_range,
        y_axis_range,
    ) = load_experiment_data(
        experiment_path,
        task_dirs,
        tasks_pool_dir,
        seed_task_dir,
    )

    st.title("Task Evolution Analysis")
    st.write(f"**Experiment:** {experiment_name}")

    ### Add the adaptation types pie chart #########################
    st.header("Adaptation Types")
    show_adaptation_types_pie_chart(task_dirs)

    ### Add the new line graph visualization #######################
    st.header("Adaptation Types Over Generations")
    show_adaptation_types_line_graph(task_dirs, G, tasks_pool_dir)

    ### Add the tSNE plot for the relevant generation ##############
    st.header(f"Task Embedding Evolution")
    if G_EMBEDDINGS is not None:
        adaptation_gens, gen_step_size = get_adaptation_generations_and_gen_step_size(
            load_generation_files(tasks_pool_dir)
        )

        # Create the allowed values for the slider: -1 (for "All") and multiples of gen_step_size starting from 1
        allowed_values = [-1] + adaptation_gens

        # slider to select the generation, with -1 representing "All"
        relevant_gen = st.select_slider(
            "Select Generation (or -1 for all tasks)",
            options=allowed_values,
            value=adaptation_gens[0],
            help="Select -1 to view all generations, or a specific generation number",
        )

        show_task_embedding_evolution_per_gen(
            task_name_to_tSNE_embedding=task_name_to_tSNE_embedding,
            task_name_to_hdbscan_cluster=task_name_to_hdbscan_cluster,
            experiment_path=experiment_path,
            relevant_gen=relevant_gen,
            all_tasks_graph=G_EMBEDDINGS,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
        )
    else:
        st.error("No tSNE embedding and HDBSCAN clusters found.")

    ### Add the evolution tree visualization #######################
    show_evolution_tree(
        task_dirs=task_dirs,
        G=G,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
