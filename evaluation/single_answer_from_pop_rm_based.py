import os
import json
import glob
import argparse
from typing import List, Any
import json
import yaml
import logging

from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.utils import remove_chat_template_from_question
from evaluation.single_answer_from_pop_analysis import (
    compute_accuracy_of_single_ans_from_pop,
    save_data_to_file,
)

# Set up basic logging to output to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to terminal
    ],
)


def evaluate_response_with_reward_model(
    prompts, responses, tokenizer, reward_model, device
) -> List[float]:
    """
    Evaluate a response using a reward model and return a scalar score.

    Args:
        prompts (List[str]): The user's input prompt
        responses (List[str]): The assistant's response to evaluate
        tokenizer: The tokenizer used for the reward model
        reward_model: The reward model for scoring responses
        device: The device (CPU/GPU) to run inference on

    Returns:
        List[float]: The list of scalar reward scores for the responses
    """
    # Create conversation in the expected format
    formatted_conversations = []
    for prompt, response in zip(prompts, responses):
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Format the conversation using the chat template
        formatted_conv = tokenizer.apply_chat_template(
            conversation, tokenize=False
        )

        # Remove potential duplicate BOS token
        if tokenizer.bos_token is not None and formatted_conv.startswith(
            tokenizer.bos_token
        ):
            formatted_conv = formatted_conv[len(tokenizer.bos_token) :]

        formatted_conversations.append(formatted_conv)

    # Tokenize the conversation
    tokenized_conv = tokenizer(
        formatted_conversations,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)

    # Get the reward score
    with torch.no_grad():
        scores = reward_model(**tokenized_conv).logits.flatten().tolist()

    return scores


def already_evaluated(eval_details: dict, rm_name: str) -> bool:
    """
    Check if all the samples in the eval details have already been evaluated with the given reward model.
    """
    is_evaluated = True
    for sample_name, sample_details in eval_details.items():
        if (
            "rm_score" not in sample_details
            or rm_name not in sample_details["rm_score"]
        ):
            is_evaluated = False
            break

    return is_evaluated


def assign_scores_to_responses_in_eval_details(
    path_to_model_eval_details: str,
    rm: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 32,
):
    """
    Assign scores to responses in the eval details based on the reward model.
    Save the scores to the eval details file under the key `rm_score` and the reward model name.

    Args:
        path_to_model_eval_details: The path to the model eval details
        rm: The reward model
        tokenizer: The tokenizer
        device: The device
        batch_size: The batch size
    """
    # Load the eval details
    with open(path_to_model_eval_details, "r") as f:
        eval_details = json.load(f)

    rm_name = rm.name_or_path.split("/")[-1]

    if already_evaluated(eval_details, rm_name):
        print(f"Already evaluated {rm_name} for {path_to_model_eval_details}")
        return eval_details

    # Assign scores to responses
    prompts = []
    responses = []
    for sample_name, sample_details in eval_details.items():
        prompt = remove_chat_template_from_question(sample_details["problem"])
        prompts.append(prompt)
        responses.append(sample_details["generation"])

    # Batch the prompts and responses
    batch_prompts = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]
    batch_responses = [
        responses[i : i + batch_size]
        for i in range(0, len(responses), batch_size)
    ]

    # Evaluate the responses in batches
    print(f"Evaluating {rm_name} for {path_to_model_eval_details}...")
    pbar = tqdm(
        zip(batch_prompts, batch_responses),
        total=len(batch_prompts),
        desc=f"Computing {rm_name} scores (bs={batch_size})...",
    )
    scores = []
    for batch_prompt, batch_response in pbar:
        batch_scores = evaluate_response_with_reward_model(
            prompts=batch_prompt,
            responses=batch_response,
            tokenizer=tokenizer,
            reward_model=rm,
            device=device,
        )
        scores.extend(batch_scores)

    for (sample_name, sample_details), score in zip(
        eval_details.items(), scores
    ):
        if "rm_score" not in sample_details:
            sample_details["rm_score"] = {}
        sample_details["rm_score"][rm_name] = score

    # Save the eval details
    with open(path_to_model_eval_details, "w") as f:
        json.dump(eval_details, f)

    return eval_details


def get_relevant_model_details(
    all_model_eval_details_paths: List[str], model_names: List[str]
) -> List[str]:
    """
    Get the relevant model details from the eval details.
    """
    relevant_model_details = []
    model_names = set(model_names)
    for model_eval_details_path in all_model_eval_details_paths:
        current_model_name = model_eval_details_path.split("/")[-1].split(
            "_eval_details.json"
        )[0]
        if current_model_name in model_names:
            relevant_model_details.append(model_eval_details_path)

    return relevant_model_details


def get_single_ans_from_pop_results(
    model_name_to_eval_details: dict, rm_name: str
) -> dict:
    """
    Get the single answer for each sample from the population.
    """

    # First, adjust dict to map from sample_name to model_details
    sample_name_to_model_details: dict[str, dict[str, dict]] = {}
    for model_name, eval_details in model_name_to_eval_details.items():
        for sample_name, sample_details in eval_details.items():
            if sample_name not in sample_name_to_model_details:
                sample_name_to_model_details[sample_name] = {}
            sample_name_to_model_details[sample_name][
                model_name
            ] = sample_details

    single_ans_from_pop_results: list[dict[str, Any]] = []
    for sample_name, all_model_details in sample_name_to_model_details.items():
        # Get the model with the highest rm score
        highest_rm_score_model_name = max(
            all_model_details,
            key=lambda x: all_model_details[x]["rm_score"][rm_name],
        )
        selected_sample_details = all_model_details[highest_rm_score_model_name]
        # Add the model name to the sample details
        selected_sample_details["model_name"] = highest_rm_score_model_name
        single_ans_from_pop_results.append(selected_sample_details)

    return single_ans_from_pop_results


def full_eval_single_ans_from_pop_rm_based(
    all_benchmark_dirs: List[str],
    model_names: List[str],
    rm: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 64,
) -> dict:
    rm_name = rm.name_or_path.split("/")[-1]

    results = {}
    for benchmark_dir in all_benchmark_dirs:
        all_model_eval_details_paths = glob.glob(
            os.path.join(benchmark_dir, "*_eval_details.json")
        )

        relevant_model_eval_details_paths = get_relevant_model_details(
            all_model_eval_details_paths, model_names
        )

        # Evaluate the responses in the eval details with the reward model
        model_name_to_eval_details = {}
        for model_eval_details_path in relevant_model_eval_details_paths:
            model_name = model_eval_details_path.split("/")[-1].split(
                "_eval_details.json"
            )[0]

            model_name_to_eval_details[model_name] = (
                assign_scores_to_responses_in_eval_details(
                    path_to_model_eval_details=model_eval_details_path,
                    rm=rm,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=batch_size,
                )
            )

        # Get the single answer for each sample from the population
        single_ans_from_pop_results = get_single_ans_from_pop_results(
            model_name_to_eval_details=model_name_to_eval_details,
            rm_name=rm_name,
        )

        # Compute the accuracy of the selected model
        accuracy, model_to_count = compute_accuracy_of_single_ans_from_pop(
            single_ans_from_pop_results,
            acc_key=None,
            filter_value=None,
        )

        results[benchmark_dir.split("/")[-1]] = {
            rm_name: accuracy,
            f"model_distribution_{rm_name}": model_to_count,
        }

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", "-e", type=str, required=True)
    parser.add_argument("--n_models", "-n", type=int, default=8)
    parser.add_argument(
        "--task_force_selection_method",
        "-t",
        type=str,
        default="global_skill_vector_coverage",
    )
    parser.add_argument(
        # "--benchmarks_file", "-b", type=str, default="benchmarks_main.yaml"
        "--benchmarks_file",
        "-b",
        type=str,
        # default="benchmarks_all.yaml",
        default="benchmarks_code.yaml",
    )
    parser.add_argument(
        "--rm_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    parser.add_argument("--overwrite_json_files", action="store_true")
    parser.add_argument("--baseline_group", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    ### Construct the important paths ##################################
    path_to_model_eval_details = os.path.join(
        args.experiment_path, "eval", "model_eval_details"
    )
    path_to_save_dir = os.path.join(
        args.experiment_path, "eval", "single_answer_from_pop"
    )

    ### Load model and tokenizer #######################################
    device = args.device
    model_name = args.rm_name
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ### Load the model names from experiment dir #######################
    if args.baseline_group is not None:
        assert args.baseline_group in [
            "experts",
            "control",
        ], "Baseline group must be either 'experts' or 'control'"

        # Get the dir for the first benchmark to all selection methods
        path_to_first_benchmark_dir = glob.glob(
            os.path.join(args.experiment_path, "coverage", "*")
        )[0]

        if args.baseline_group == "experts":
            results_json_name = f"results_N{args.n_models}.json"
        elif args.baseline_group == "control":
            results_json_name = f"control_results_N{args.n_models}.json"
        path_to_results_json = os.path.join(
            path_to_first_benchmark_dir, results_json_name
        )

        model_group_name = f"{args.baseline_group}_N{args.n_models}"
    else:
        # Get the dir for the first benchmark to all selection methods
        path_to_first_benchmark_dir = glob.glob(
            os.path.join(args.experiment_path, "eval", "coverage", "*")
        )[0]
        # Get the top N models for the specified selection method
        path_to_results_json = os.path.join(
            path_to_first_benchmark_dir,
            args.task_force_selection_method,
            f"results_N{args.n_models}.json",
        )

        model_group_name = f"N{args.n_models}"
    with open(path_to_results_json, "r") as f:
        selection_method_results = json.load(f)
    model_names = list(
        selection_method_results["coverage_contributions"].keys()
    )

    ### Get the benchmark dirs #########################################
    all_benchmark_dirs = glob.glob(
        os.path.join(path_to_model_eval_details, "*")
    )

    base_path_to_config = "evaluation/single_answer_selection_configs"
    with open(f"{base_path_to_config}/{args.benchmarks_file}", "r") as f:
        benchmarks_to_keep = yaml.safe_load(f)

    all_benchmark_dirs = [
        benchmark_dir
        for benchmark_dir in all_benchmark_dirs
        if benchmark_dir.split("/")[-1] in benchmarks_to_keep
    ]

    ### Evaluate the single answer from the population #################
    results = full_eval_single_ans_from_pop_rm_based(
        all_benchmark_dirs=all_benchmark_dirs,
        model_names=model_names,
        rm=rm,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )

    ### Save the results ###############################################
    save_data_to_file(
        data=results,
        path_to_save_dir=path_to_save_dir,
        model_group_name=model_group_name,
        overwrite_json_files=args.overwrite_json_files,
        task_force_selection_method=args.task_force_selection_method,
    )


if __name__ == "__main__":
    main()
