"""
This script computes the Vendi Score for a given dataset.

1. Compute the embedding of each sample in the dataset (if not already computed) and store it in the metadata.json file
2. Compute a similarity matrix K (N x N)
3. Compute the Vendi Score

The script gets as input the path to the task archive.
From there, it (1) computes the embedding of each sample in the dataset.
Then, it performs (2) and (3) for each task pool generation in the archive.

The files are stored in the following structure:
<archive_path>/
|-- <task_name_1>/
|   |-- metadata.json
|   |-- task.json
|   |-- task.py
|-- <task_name_2>/
|   |-- metadata.json
|   |-- task.json
|   |-- task.py
|-- ...
|-- activate_pool_gen_<X>.json

Where `activate_pool_gen_<X>.json` is a json file with the following structure:

[
    "path/to/task_name_1",
    "path/to/task_name_2",
    ...
    "path/to/task_name_N",
]

And `metadata.json` is a json file with the following structure:

{
    "generated_from_task": <task_name_X>,
    "task_number": <task_number_X>,
    "generation_type": <generation_type_X>,
    "original_pass_rate": <original_pass_rate_X>
}

And `task.json` is a json file with the following structure:

{
    "name_of_task": <name_of_task_X>,
    "description_of_task": <description_of_task_X>,
    "capability_being_measured": <capability_being_measured_X>,
    "estimated_human_difficulty": <estimated_human_difficulty_X>,
    "example_instruction": <example_instruction_X>
}

The script saves a json file which stores the Vendi Score for each task pool generation with the following structure:
It saves it at archive_path/vendi_scores_per_generation.json

{
    "0": <vendi_score>
    "1": <vendi_score>
    ...
    "N": <vendi_score>
}
"""

from typing import Any, Dict
import numpy as np
import logging
import os
import json
from glob import glob
import argparse

from openai import OpenAI
from vendi_score import vendi


logger = logging.getLogger(__name__)

# TODO: Change to use config file and/or environment variables
embedding_model_config = {
    "embedding_model_name": "intfloat/e5-mistral-7b-instruct",  # ~27,655 MiB GPU memory
    # embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2" # ~820 MiB GPU memory
    "vllm_embedding_server_host": "172.16.15.207",  # Host where vLLM server is running
    "vllm_embedding_server_port": 8010,  # Port where vLLM server is running
    "embedding_vllm_url": """http://{vllm_embedding_server_host}:{vllm_embedding_server_port}/v1""",  # Base URL for vLLM OpenAI API
}

# Use environment variable to get embedding model config


def get_enbedding_model_client(embedding_model_config):
    """
    Get the embedding model client.
    """
    embedding_vllm_url = embedding_model_config["embedding_vllm_url"].format(
        vllm_embedding_server_host=embedding_model_config["vllm_embedding_server_host"],
        vllm_embedding_server_port=embedding_model_config["vllm_embedding_server_port"],
    )
    client = OpenAI(
        base_url=embedding_vllm_url,
        api_key="dummy API key",
    )
    return client


def get_task_text_representation(task_description_dict: Dict[str, Any]) -> str:
    """
    Get the task text representation from the task description dictionary.
    """
    text_to_embed = ""
    for key, value in task_description_dict.items():
        text_to_embed += f"{key}: {value}\n"
    return text_to_embed


def embed_text(
    embedding_model_client: OpenAI, embedding_model_name: str, text: str
) -> np.ndarray:
    """
    Generate embeddings for the given text using the configured OpenAI client.

    Args:
        text: The text to embed.

    Returns:
        A numpy array containing the embedding vector.
    """
    try:
        embedding_data = (
            embedding_model_client.embeddings.create(
                input=text,
                model=embedding_model_name,
            )
            .data[0]
            .embedding
        )
        return np.array(embedding_data, dtype=np.float32)
    except Exception as e:
        raise e
        # return np.zeros(
        #     self.dimension, dtype=np.float32
        # )  # Or return zero vector


def get_task_embedding_from_vector_db(
    task_name: str, vector_db_path: str
) -> np.ndarray:
    """
    Get the embedding of a task from the vector database.
    """
    embedding_path = os.path.join(vector_db_path, task_name + ".npy")
    return np.load(embedding_path)


def compute_similarity_matrix(
    embeddings: list[np.ndarray],
) -> np.ndarray:
    """
    Compute the cosine similarity matrix for a list of embeddings.
    """
    if len(embeddings) == 0:
        return np.array([])
    emb_matrix = np.stack(embeddings)
    norm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norm + 1e-8)
    return np.dot(emb_matrix, emb_matrix.T)


def update_to_local_experiment_path(task_path: str, experiment_path: str):
    """
    Update the task path to the local experiment path.
    """
    # only keep the last 3 parts of the task path
    task_path = "/".join(task_path.split("/")[-3:])
    return os.path.join(experiment_path, task_path)


def get_vendi_scores(archive_path: str):
    """
    Compute the Vendi Score for each pool generation in the archive.

    Args:
        archive_path: Path to the task archive.

    Returns:
        local_vendi_scores: Dictionary mapping generation number to local Vendi Score.
            Local means the Vendi Score is computed for the tasks in the current generation.
        global_vendi_scores: Dictionary mapping generation number to global Vendi Score.
            Global means the Vendi Score is computed for all tasks in the archive up to the current generation.
    """

    # 1. For each pool generation, compute Vendi Score
    logger.info(f"Computing Vendi Score for each pool generation in {archive_path}...")
    local_vendi_scores = {}
    global_vendi_scores = {}

    experiment_path = os.path.dirname(os.path.dirname(archive_path))

    # Use dictionary to make sure we don't store the same embeddings multiple times
    all_embeddings = {}
    gen_files = glob(os.path.join(archive_path, "active_pool_gen_*.json"))
    gen_files.sort(key=lambda x: int(x.split("gen_")[-1].split(".")[0]))
    for gen_file in gen_files:
        gen_idx = int(gen_file.split("gen_")[-1].split(".")[0])
        with open(gen_file, "r") as f:
            task_paths = json.load(f)
        embeddings = []
        for abs_task_path in task_paths:
            abs_task_path = update_to_local_experiment_path(
                abs_task_path, experiment_path
            )
            metadata_path = os.path.join(abs_task_path, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            embedding = np.array(metadata["embedding"], dtype=np.float32)
            embeddings.append(embedding)
            all_embeddings[metadata["task_number"]] = embedding

        # Compute local Vendi Score
        K = compute_similarity_matrix(embeddings)
        vendi_score = vendi.score_K(K)
        local_vendi_scores[gen_idx] = float(vendi_score)

        # Compute global Vendi Score
        embeddings = list(all_embeddings.values())
        K = compute_similarity_matrix(embeddings)
        vendi_score = vendi.score_K(K, q=0.5)
        global_vendi_scores[gen_idx] = float(vendi_score)

    # 2. Save results
    try:
        # local Vendi Scores
        logger.info(
            "Saving Vendi Scores for each pool generation under "
            f"{os.path.join(archive_path, 'vendi_scores_per_generation.json')}..."
        )
        with open(
            os.path.join(archive_path, "vendi_scores_per_generation.json"), "w"
        ) as f:
            json.dump(local_vendi_scores, f, indent=2)

        # global Vendi Scores
        logger.info(
            "Saving Vendi Scores for each pool generation under "
            f"{os.path.join(archive_path, 'global_vendi_scores_per_generation.json')}..."
        )
        with open(
            os.path.join(archive_path, "global_vendi_scores_per_generation.json"),
            "w",
        ) as f:
            json.dump(global_vendi_scores, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving Vendi scores: {e}")

    return local_vendi_scores, global_vendi_scores


def compute_vendi_scores(archive_path):
    # archive_path/../../..
    experiment_path = os.path.dirname(os.path.dirname(os.path.dirname(archive_path)))
    embedding_model_client = get_enbedding_model_client(embedding_model_config)
    embedding_model_name = embedding_model_config["embedding_model_name"]
    embedding_model_name = (
        embedding_model_name.split("/")[-1]
        if "/" in embedding_model_name
        else embedding_model_name
    )

    # 1. Compute/store embeddings for all tasks
    logger.info(f"Computing all missing embeddings for all tasks in {archive_path}...")
    task_paths = glob(os.path.join(archive_path, "task_*"))
    for task_path in task_paths:
        metadata_path = os.path.join(task_path, "metadata.json")
        task_json_path = os.path.join(task_path, "task.json")

        # Check for embedding
        if not os.path.exists(metadata_path) or not os.path.exists(task_json_path):
            continue
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        task_embedding = metadata.get("embedding", None)

        # If embedding is not present, compute it
        if not task_embedding:
            with open(task_json_path, "r") as f:
                task_json = json.load(f)
            text = get_task_text_representation(task_json)
            # embedding = embed_text(
            #     embedding_model_client, embedding_model_name, text
            # )
            embedding = get_task_embedding_from_vector_db(
                task_name=os.path.basename(task_path),
                vector_db_path=os.path.join(
                    experiment_path,
                    "vector_db_historical",
                    "vectors",
                ),
            )
            metadata["embedding"] = embedding.tolist()
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    # 2. Compute Vendi Scores
    local_vendi_scores, global_vendi_scores = get_vendi_scores(archive_path)

    logger.info("Done!")
    return local_vendi_scores, global_vendi_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Vendi scores for task pools")
    parser.add_argument(
        "--archive_path",
        "-p",
        type=str,
        default=None,
        required=True,
        help="Path to the task archive",
    )
    args = parser.parse_args()

    if args.archive_path is None:
        raise ValueError("archive_path is required")

    compute_vendi_scores(args.archive_path)
