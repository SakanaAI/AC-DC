import os
import re
import shutil
import json
import torch
import vllm
import numpy as np
import pandas as pd
import matplotlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import asdict
from collections import defaultdict
import logging
import glob
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def state_dict_hf_to_vllm_qwen(state_dict: dict) -> dict:
    """
    Convert a state dict from a Hugging Face model to a VLLM model.
    """
    new_state_dict = {}

    for key in state_dict:
        if "self_attn" in key:
            # Handle attention weights
            layer_prefix = key[: key.rfind("self_attn") + len("self_attn")]
            if "q_proj" in key or "k_proj" in key or "v_proj" in key:
                # Collect q, k, v weights/biases to combine them
                if "weight" in key:
                    q = state_dict[f"{layer_prefix}.q_proj.weight"]
                    k = state_dict[f"{layer_prefix}.k_proj.weight"]
                    v = state_dict[f"{layer_prefix}.v_proj.weight"]
                    qkv = torch.cat([q, k, v], dim=0)
                    new_state_dict[f"{layer_prefix}.qkv_proj.weight"] = qkv
                elif "bias" in key:
                    q = state_dict[f"{layer_prefix}.q_proj.bias"]
                    k = state_dict[f"{layer_prefix}.k_proj.bias"]
                    v = state_dict[f"{layer_prefix}.v_proj.bias"]
                    qkv = torch.cat([q, k, v], dim=0)
                    new_state_dict[f"{layer_prefix}.qkv_proj.bias"] = (
                        qkv  # This key only exists if bias=True in config
                    )
            # elif "o_proj" in key:  # Explicitly handle o_proj
            #     new_state_dict[key] = state_dict[key]
            # elif "k_norm" in key or "q_norm" in key:
            #     new_state_dict[key] = state_dict[key]
            # catch all other keys like o_proj, q_norm (qwen3), k_norm (qwen3), etc.
            else:
                new_state_dict[key] = state_dict[key]
        elif "mlp" in key:
            # Handle MLP weights
            layer_prefix = key[: key.rfind("mlp") + len("mlp")]
            if "gate_proj" in key or "up_proj" in key:
                if "weight" in key:
                    gate = state_dict[f"{layer_prefix}.gate_proj.weight"]
                    up = state_dict[f"{layer_prefix}.up_proj.weight"]
                    gate_up = torch.cat([gate, up], dim=0)
                    new_state_dict[f"{layer_prefix}.gate_up_proj.weight"] = (
                        gate_up
                    )
            elif "down_proj" in key:  # Explicitly handle down_proj
                new_state_dict[key] = state_dict[key]
        else:
            # Copy other weights as is
            new_state_dict[key] = state_dict[key]

    if "lm_head.weight" not in new_state_dict:
        logger.warning(
            "lm_head.weight not found in state_dict. Assuming word embedding tying and using input embeddings as output embeddings. This is expected for Qwen 1.5B models."
        )
        new_state_dict["lm_head.weight"] = new_state_dict[
            "model.embed_tokens.weight"
        ]

    return new_state_dict


def state_dict_hf_to_vllm_llama(state_dict: dict) -> dict:
    """
    Convert a Llama state dict from a Hugging Face model to a VLLM model.
    """
    new_state_dict = {}

    for key in state_dict:
        if "self_attn" in key:
            # Handle attention weights
            layer_prefix = key[: key.rfind("self_attn") + len("self_attn")]
            if "q_proj" in key or "k_proj" in key or "v_proj" in key:
                # Collect q, k, v weights to combine them
                if "weight" in key:
                    q = state_dict[f"{layer_prefix}.q_proj.weight"]
                    k = state_dict[f"{layer_prefix}.k_proj.weight"]
                    v = state_dict[f"{layer_prefix}.v_proj.weight"]
                    qkv = torch.cat([q, k, v], dim=0)
                    new_state_dict[f"{layer_prefix}.qkv_proj.weight"] = qkv
            elif "o_proj" in key:  # Explicitly handle o_proj
                new_state_dict[key] = state_dict[key]
        elif "mlp" in key:
            # Handle MLP weights
            layer_prefix = key[: key.rfind("mlp") + len("mlp")]
            if "gate_proj" in key or "up_proj" in key:
                if "weight" in key:
                    gate = state_dict[f"{layer_prefix}.gate_proj.weight"]
                    up = state_dict[f"{layer_prefix}.up_proj.weight"]
                    gate_up = torch.cat([gate, up], dim=0)
                    new_state_dict[f"{layer_prefix}.gate_up_proj.weight"] = (
                        gate_up
                    )
            elif "down_proj" in key:  # Explicitly handle down_proj
                new_state_dict[key] = state_dict[key]
        else:
            # Copy other weights as is
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def load_safetensors_state_dict(index_path: str) -> dict:
    """
    Load a complete state dict from safetensors files using an index file.

    Args:
        index_path: Path to the model.safetensor.index.json file

    Returns:
        Complete state dictionary containing all model weights
    """
    # Initialize empty state dict
    state_dict = {}

    # Get all safetensors files in the directory
    all_weight_files = glob.glob(os.path.join(index_path, "*.safetensors"))

    for weight_file in all_weight_files:
        # load the weight file
        state_dict_shard = load_file(weight_file)
        # update the state dict
        state_dict.update(state_dict_shard)

    return state_dict


def update_vllm_weights(param: Dict, llm: vllm.LLM):
    """
    Update the VLLM model weights with the weights from the local file.

    Args:
        llm: The VLLM model to update.
        local_weights_path: The path to the local safetensors weights file.
    """

    ### Load the weights
    # if local_weights_path.endswith(".safetensors"):
    #     state_dict = load_file(local_weights_path)
    # else:
    #     # Fallback to original loading method for other formats
    #     state_dict = AutoModelForCausalLM.from_pretrained(
    #         local_weights_path, torch_dtype=torch.bfloat16
    #     ).state_dict()

    # Create a new state dict with adjusted state dict
    new_state_dict = state_dict_hf_to_vllm_qwen(param)

    def update_vllm_weights_func(model):
        # strict=False ONLY because the lm_head.weight is not present in the new_state_dict
        # Ensure that all other keys are present in the new_state_dict!!!
        model.load_state_dict(new_state_dict, strict=True)

    try:
        llm.apply_model(update_vllm_weights_func)
    except Exception as e:
        raise Exception(f"Failed to update VLLM weights: {e}")


def update_vllm_weights_general(
    param: Dict,
    llm: vllm.LLM,
    hf_to_vllm_conversion_fn: Callable,
):
    """
    Update the VLLM model weights with the weights from the local file.

    Args:
        llm: The VLLM model to update.
        local_weights_path: The path to the local safetensors weights file.
    """

    def update_vllm_weights_func(model):
        model.load_state_dict(hf_to_vllm_conversion_fn(param), strict=True)

    try:
        llm.apply_model(update_vllm_weights_func)
    except Exception as e:
        raise Exception(f"Failed to update VLLM weights: {e}")


def load_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    """
    Load parameters from a Hugging Face model state dict into a vLLM model.

    This function transfers weights from a standard Hugging Face PyTorch model
    state dictionary to the corresponding vLLM model parameters. It handles the
    differences in parameter organization between the two formats, including
    concatenating separate Q, K, V projection matrices into vLLM's combined format.

    Args:
        param: Dictionary containing Hugging Face model parameters
        llm: vLLM model instance to load parameters into

    Note:
        This function expects the vLLM model architecture to be compatible with
        the Hugging Face model architecture (e.g., same number of layers, same
        hidden dimensions, etc.)
    """
    # model = llm.llm_engine.driver_worker.model_runner.model  # only for vllm 0.3.0
    model = (
        llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    )
    num_layers = model.config.num_hidden_layers

    # Load embeddings layer weights.
    model_param = model.get_parameter("model.embed_tokens.weight")
    model_param.copy_(
        param["model.embed_tokens.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )
    model_param = model.get_parameter("lm_head.weight")
    model_param.copy_(
        param["lm_head.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )

    # Load the final layernorm weights.
    model_param = model.get_parameter("model.norm.weight")
    model_param.copy_(
        param["model.norm.weight"].to(model_param.dtype).to(model_param.device)
    )

    for i in range(num_layers):
        # Load qkv_proj weights.
        model_param = model.get_parameter(
            f"model.layers.{i}.self_attn.qkv_proj.weight"
        )
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.self_attn.q_proj.weight"],
                    param[f"model.layers.{i}.self_attn.k_proj.weight"],
                    param[f"model.layers.{i}.self_attn.v_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load gate_up_proj weights.
        model_param = model.get_parameter(
            f"model.layers.{i}.mlp.gate_up_proj.weight"
        )
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.mlp.gate_proj.weight"],
                    param[f"model.layers.{i}.mlp.up_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load o_proj and down_proj weights.
        model_param = model.get_parameter(
            f"model.layers.{i}.self_attn.o_proj.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.self_attn.o_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(
            f"model.layers.{i}.mlp.down_proj.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.mlp.down_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load layer_norm weights.
        model_param = model.get_parameter(
            f"model.layers.{i}.input_layernorm.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.input_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(
            f"model.layers.{i}.post_attention_layernorm.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.post_attention_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )


def load_qwen_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    """
    Load parameters from a Hugging Face model state dict into a vLLM model.

    This function transfers weights from a standard Hugging Face PyTorch model
    state dictionary to the corresponding vLLM model parameters. It handles the
    differences in parameter organization between the two formats, including
    concatenating separate Q, K, V projection matrices into vLLM's combined format.

    Args:
        param: Dictionary containing Hugging Face model parameters
        llm: vLLM model instance to load parameters into

    Note:
        This function expects the vLLM model architecture to be compatible with
        the Hugging Face model architecture (e.g., same number of layers, same
        hidden dimensions, etc.)
    """
    model = (
        llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    )
    num_layers = model.config.num_hidden_layers

    # Load embeddings layer weights.
    model_param = model.get_parameter("model.embed_tokens.weight")
    model_param.copy_(
        param["model.embed_tokens.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )

    # Load the final layernorm weights.
    model_param = model.get_parameter("model.norm.weight")
    model_param.copy_(
        param["model.norm.weight"].to(model_param.dtype).to(model_param.device)
    )

    for i in range(num_layers):
        # Load qkv_proj weights and biases.
        # Weights
        model_param_w = model.get_parameter(
            f"model.layers.{i}.self_attn.qkv_proj.weight"
        )
        model_param_w.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.self_attn.q_proj.weight"],
                    param[f"model.layers.{i}.self_attn.k_proj.weight"],
                    param[f"model.layers.{i}.self_attn.v_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param_w.dtype)
            .to(model_param_w.device)
        )
        # Biases (Qwen specific)
        model_param_b = model.get_parameter(
            f"model.layers.{i}.self_attn.qkv_proj.bias"
        )
        model_param_b.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.self_attn.q_proj.bias"],
                    param[f"model.layers.{i}.self_attn.k_proj.bias"],
                    param[f"model.layers.{i}.self_attn.v_proj.bias"],
                ],
                dim=0,
            )
            .to(model_param_b.dtype)
            .to(model_param_b.device)
        )
        # Load gate_up_proj weights.
        model_param = model.get_parameter(
            f"model.layers.{i}.mlp.gate_up_proj.weight"
        )
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.mlp.gate_proj.weight"],
                    param[f"model.layers.{i}.mlp.up_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load o_proj and down_proj weights.
        model_param = model.get_parameter(
            f"model.layers.{i}.self_attn.o_proj.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.self_attn.o_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(
            f"model.layers.{i}.mlp.down_proj.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.mlp.down_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load layer_norm weights.
        model_param = model.get_parameter(
            f"model.layers.{i}.input_layernorm.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.input_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(
            f"model.layers.{i}.post_attention_layernorm.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.post_attention_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )


def save_archive_map(data_map: Dict, output_path: str) -> None:
    serializable_map = {}
    for outer_key, inner_dict in data_map.items():
        serializable_inner_dict = {}
        for tuple_key, archive_data in inner_dict.items():
            string_key = ",".join(map(str, tuple_key))
            serializable_inner_dict[string_key] = asdict(archive_data)
        serializable_map[outer_key] = serializable_inner_dict

    with open(output_path, "w") as f:
        json.dump(serializable_map, f, indent=4)


def load_archive_map(
    input_path: str, archive_data_class: Any
) -> Dict[Any, Dict[Tuple, Any]]:
    with open(input_path, "r") as f:
        serializable_map = json.load(f)

    data_map = {}
    for outer_key, inner_dict in serializable_map.items():
        deserialized_inner_dict = {}
        for string_key, archive_data_dict in inner_dict.items():
            tuple_key = tuple(map(int, string_key.split(",")))
            archive_data = archive_data_class(**archive_data_dict)
            deserialized_inner_dict[tuple_key] = archive_data
        data_map[outer_key] = deserialized_inner_dict

    return data_map


def delete_outdated_models(
    data_map: Dict, model_dir: str, threshold: int
) -> List[str]:
    """Delete outdated models from QD archive."""
    model_path_in_archive = {
        os.path.basename(data_map[k][bc_ids].model_path)
        for k in data_map
        for bc_ids in data_map[k]
    }
    model_path_on_disk = os.listdir(model_dir)
    deleted_models = []
    for model_path in model_path_on_disk:
        # Extract generation number from gen_X_ind_Y format
        try:
            # Find the part that starts with gen_
            gen_part = [
                part for part in model_path.split("_") if part.isdigit()
            ][0]
            gen_id = int(gen_part)
        except (IndexError, ValueError):
            # If we can't parse the generation number, skip this model
            continue

        # Delete the model if it is not in the archive map and not generated
        # by a worker that just finished its work but has not been dealt with.
        if model_path not in model_path_in_archive and gen_id < threshold:
            full_model_path = os.path.join(model_dir, model_path)
            if os.path.exists(full_model_path):
                try:
                    shutil.rmtree(full_model_path)
                    deleted_models.append(full_model_path)
                except OSError as e:
                    print(f"Error deleting {full_model_path}: {e}")
    return deleted_models


def delete_models_not_in_archive(
    model_dir: str,
    keep_model_paths: List[str],
    threshold: int,
    skip_interval: Optional[int] = None,
) -> List[str]:
    """Delete models that are not in the DNS/BDMA archive and older than threshold.

    Args:
        model_dir: Directory containing model files
        keep_model_paths: List of model paths to keep (from DNS/BDMA archive)
        threshold: Generation number threshold - delete models older than this
        skip_interval: If set, do not delete models for every Nth generation (keep all models for generations that are multiples of this interval, except 0)

    Returns:
        List of deleted model paths
    """
    # Convert archive model paths to basenames for comparison
    keep_models = {os.path.basename(path) for path in keep_model_paths}

    # Get all models on disk
    model_path_on_disk = os.listdir(model_dir)
    deleted_models = []

    for model_path in model_path_on_disk:
        # Extract generation number from gen_X_ind_Y format
        try:
            gen_part = [
                part for part in model_path.split("_") if part.isdigit()
            ][0]
            gen_id = int(gen_part)
        except (IndexError, ValueError):
            # Skip if we can't parse the generation number
            continue

        # Don't remove the parent model mapping directory
        if model_path == "parent_models_mapping":
            continue

        # Skip deletion for every Nth generation if skip_interval is set (except 0)
        if (
            skip_interval is not None
            and skip_interval > 0
            # and gen_id != 0
            and gen_id % skip_interval == 0
        ):
            continue
        # Delete if model is not in keep list and older than threshold
        if model_path not in keep_models and gen_id < threshold:
            full_model_path = os.path.join(model_dir, model_path)
            if os.path.exists(full_model_path):
                try:
                    shutil.rmtree(full_model_path)
                    deleted_models.append(full_model_path)
                except OSError as e:
                    print(f"Error deleting {full_model_path}: {e}")

    return deleted_models


def cleanup_old_models(cfg, gen, archive_data):
    """Clean up old model files based on optimization mode, supports BDMA and DNS.

    Args:
        cfg: Hydra configuration object
        gen: Current generation
        archive_data: Archive data
    """
    model_dir = archive_data["dirs"]["model_dir"]

    if cfg.run_bdma:
        model_paths = [
            sol.model_path for sol in archive_data["bdma_archive"].solutions
        ]
    elif cfg.run_dns:
        model_paths = [sol.model_path for sol in archive_data["dns_archive"]]
    else:
        return delete_outdated_models(
            data_map=archive_data["archive_map"],
            model_dir=model_dir,
            threshold=gen,
        )

    return delete_models_not_in_archive(
        model_dir=model_dir, keep_model_paths=model_paths, threshold=gen
    )


def get_largest_gen_file(model_dir):
    # List all files in the directory
    files = os.listdir(model_dir)

    # Regular expression to match the generation files and extract generation number
    gen_regex = re.compile(r"gen(\d+)_archive_map\.json")

    # Extract generation numbers and their corresponding files
    gen_files = {}
    for file in files:
        match = gen_regex.match(file)
        if match:
            gen_num = int(match.group(1))
            gen_files[gen_num] = file

    # Find the largest generation number and its corresponding file
    largest_gen_num = max(gen_files.keys())
    largest_gen_file = gen_files[largest_gen_num]

    return largest_gen_file, largest_gen_num


def get_latest_generation(model_dir):
    """Find the latest generation number by looking at model files.

    Args:
        model_dir: Directory containing model files with pattern gen_X_ind_Y

    Returns:
        int: The latest generation number found

    Raises:
        ValueError: If no model files found matching the expected pattern
    """
    # List all files in the directory
    files = os.listdir(model_dir)

    # Regular expression to match model files and extract generation number
    gen_regex = re.compile(r"gen_(\d+)_ind_.*")

    # Extract all generation numbers
    gen_numbers = []
    for file in files:
        match = gen_regex.match(file)
        if match:
            gen_num = int(match.group(1))
            gen_numbers.append(gen_num)

    if not gen_numbers:
        raise ValueError(
            f"No model files found matching pattern gen_X_ind_Y in {model_dir}"
        )

    # Return the largest generation number
    return max(gen_numbers)


def plot_elite_map(
    archive_map_path: str, task_configs: dict, output_path: str, data_split: str
) -> None:
    # Load the archive map.
    with open(archive_map_path, "r") as f:
        data = json.load(f)

    # Plot the elite maps
    data_len = len(data.keys())
    if data_len == 3:
        plot_num = 1
    elif data_len == 4:
        plot_num = 3
    else:
        return None
        # raise NotImplementedError(f"Data length {data_len} not supported yet.")

    df_dict = {}
    for k in data:
        df_dict[k] = {}
        bc_num_dim = len(list(data[k].keys())[0].split(","))
        for i in range(bc_num_dim):
            df_dict[k][f"bc_dim{i}"] = []
        df_dict[k]["quality"] = []
        if data_split == "validation":
            df_dict[k]["train_quality"] = []
        for bc_ids in data[k]:
            for i, bc_id in enumerate(bc_ids.split(",")):
                df_dict[k][f"bc_dim{i}"].append(int(bc_id))
            if data_split in ["all", "train"]:
                df_dict[k]["quality"].append(data[k][bc_ids]["quality"])
            elif data_split == "validation":
                df_dict[k]["quality"].append(
                    data[k][bc_ids]["validation_quality"]
                )
                df_dict[k]["train_quality"].append(data[k][bc_ids]["quality"])
            else:
                raise ValueError(f"Invalid data split: {data_split}")

    fig, axes = plt.subplots(
        plot_num, data_len, figsize=(9 * data_len, 8 * plot_num)
    )

    for i, key_q in enumerate(df_dict):
        df = pd.DataFrame(df_dict[key_q])
        q_min = task_configs[key_q]["bc_min_vals"][0]
        q_max = task_configs[key_q]["bc_max_vals"][0]
        other_keys = [key for key in task_configs.keys() if key != key_q]
        for j in range(plot_num):
            if j == 0:
                key_x_idx = 0
                key_y_idx = 1
            elif j == 1:
                key_x_idx = 1
                key_y_idx = 2
            elif j == 2:
                key_x_idx = 0
                key_y_idx = 2
            else:
                raise ValueError(f"Invalid plot index: {j}")

            ax = axes[j][i] if plot_num > 1 else axes[i]

            key_x = other_keys[key_x_idx]
            x_min = task_configs[key_x]["bc_min_vals"][0]
            x_max = task_configs[key_x]["bc_max_vals"][0]
            x_grid_size = task_configs[key_x]["bc_grid_sizes"][0]

            key_y = other_keys[key_y_idx]
            y_min = task_configs[key_y]["bc_min_vals"][0]
            y_max = task_configs[key_y]["bc_max_vals"][0]
            y_grid_size = task_configs[key_y]["bc_grid_sizes"][0]

            elite_map = np.zeros([x_grid_size, y_grid_size])
            max_q_values = {}
            for _, row in df.iterrows():
                x = int(row[f"bc_dim{key_x_idx}"])
                y = int(row[f"bc_dim{key_y_idx}"])
                if data_split in ["all", "train"]:
                    q = row["quality"]
                    if (x, y) not in max_q_values or max_q_values[(x, y)] < q:
                        max_q_values[(x, y)] = q
                        elite_map[x, y] = q
                elif data_split == "validation":
                    q = row["train_quality"]
                    if (x, y) not in max_q_values or max_q_values[(x, y)] < q:
                        max_q_values[(x, y)] = q
                        elite_map[x, y] = row["quality"]

            elite_map = elite_map.T

            im = ax.imshow(elite_map, cmap="viridis", vmin=q_min, vmax=q_max)
            ax.set_title(f"{key_q}", fontsize=20)
            ax.set_xlabel(key_x, fontsize=18)
            ax.set_ylabel(key_y, fontsize=18)
            ax.set_xticks(np.arange(-0.5, x_grid_size, 1))
            ax.set_yticks(np.arange(-0.5, y_grid_size, 1))
            ax.set_xticklabels(
                np.round(np.linspace(x_min, x_max, x_grid_size + 1), 2)
            )
            ax.set_yticklabels(
                np.round(np.linspace(y_min, y_max, y_grid_size + 1), 2)
            )
            ax.invert_yaxis()

            median_val = np.median(elite_map[elite_map > 0])
            for _, row in df.iterrows():
                x = int(row[f"bc_dim{key_x_idx}"])
                y = int(row[f"bc_dim{key_y_idx}"])
                q = elite_map[y, x]
                c = "white" if q < median_val else "red"
                ax.text(
                    x,
                    y,
                    f"{q:.2f}",
                    ha="center",
                    va="center",
                    color=c,
                    fontsize=14,
                )

            fig.colorbar(im, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def elite_model_table(
    archive_map_path: str, output_path: str, top_k: int = 8
) -> None:
    # Load the archive map.
    with open(archive_map_path, "r") as f:
        data = json.load(f)

    top_k_data = {}
    for key, value in data.items():
        # Convert to DataFrame
        df = pd.DataFrame(value).T
        df = df.sort_values(by="quality", ascending=False).head(top_k)
        top_k_data[key] = df

    def find_dict_by_model_path(data, model_path):
        for key, value in data.items():
            if value.get("model_path") == model_path:
                if value.get("validation_quality") is not None:
                    return round(value.get("validation_quality"), 3)
                else:
                    "not max model"
        return "not elite"

    result = []
    for category, df in top_k_data.items():
        for index, row in df.iterrows():
            result_dict = {
                "category": category,
                "model_name": os.path.basename(row["model_path"]),
                "training": round(row["quality"], 3),
            }
            for key in data.keys():
                key_validation = find_dict_by_model_path(
                    data[key], row["model_path"]
                )
                result_dict[key] = key_validation
            result.append(result_dict)

    # Create a DataFrame
    df = pd.DataFrame(result)

    # Check for duplicate model names and keep track of their indices
    model_name_counts = defaultdict(int)
    duplicate_indices = []

    for idx, name in enumerate(df["model_name"]):
        model_name_counts[name] += 1
        if model_name_counts[name] > 1:
            duplicate_indices.append(idx)

    # Visualize the data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for idx, cell in table.get_celld().items():
        row, col = idx
        if (
            row > 0
            and col == df.columns.get_loc("model_name")
            and row in duplicate_indices
        ):
            cell.set_text_props(color="red")

    plt.savefig(output_path)
