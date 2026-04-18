from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os
import sys
import logging
import multiprocessing
from omegaconf import DictConfig
import re
import hydra
from vllm import LLM, SamplingParams, TokensPrompt
import logging
import glob
import json
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# add path to this directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tasks.acdc_task import ACDCTask
from main_ac_dc import setup_optimization_directories
from tasks.task_generation import ACDCTaskPool
from utils.helpers import (
    load_safetensors_state_dict,
    state_dict_hf_to_vllm_llama,
    state_dict_hf_to_vllm_qwen,
)
from tasks.task_gen_prompts import (
    eval_cot_system_msg,
    eval_zs_system_msg,
)


# --- Helper functions ---
def extract_answer_from_raw_output(raw_output: str) -> str:
    """
    Extract the answer from the raw output of the LLM.
    """
    ### Look for \boxed{<answer>} in the raw_output
    answer = "No answer found."
    if re.search(r"\\boxed\{(.*)\}", raw_output):
        match = re.search(r"\\boxed\{(.*)\}", raw_output, re.DOTALL)
        if match:
            answer = match.group(1)

    # TODO: Discuss with @Andrew.
    # Remove the answer extraction based on `####` in `global_task_pool_eval`?
    # This could bias the score of the gsm8k expert, which might get arteficially good scores,
    # since we are handling the answer extraction, but on downstream benchmarks,
    # where the answer extraction is different the gsm8k expert performes very poorly.

    ### Look for "####" in the raw output
    if answer == "No answer found.":
        match = re.search(r"####(.*)", raw_output, re.DOTALL)
        if match:
            answer = match.group(1).strip()

    ### look for "Answer: <answer>" in the raw_output
    if answer == "No answer found.":
        match = re.search(r"Answer:(.*)", raw_output, re.DOTALL)
        if match:
            answer = match.group(1)
    return answer


def update_vllm_weights(param: Dict, llm: LLM):
    """
    Update the VLLM model weights with the weights from the local file.

    Args:
        llm: The VLLM model to update.
        local_weights_path: The path to the local safetensors weights file.
    """

    def update_vllm_weights_func(model):
        model.load_state_dict(param, strict=True)

    try:
        llm.apply_model(update_vllm_weights_func)
    except Exception as e:
        raise Exception(f"Failed to update VLLM weights: {e}")


def log_prompts_and_responses(
    prompts: List[str],
    responses: List[str],
    answers: List[str],
    task_dirs: List[str],
    log_file_name: str,
    acdc_skill_vector: List[float],
):
    """Log all prompts and their responses to a json file."""
    prompts_to_responses = {}
    for prompt, response, answer, task_dir in zip(
        prompts, responses, answers, task_dirs
    ):
        task_id = task_dir.split("/")[-1]
        prompts_to_responses[task_id] = {
            "prompt": prompt,
            "response": response,
            "answer": answer,
            "task_dir": task_dir,
            "score": acdc_skill_vector[task_id],
        }

    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
    with open(log_file_name, "w") as f:
        json.dump(prompts_to_responses, f, indent=2)

    return prompts_to_responses


@dataclass
class ACDCTaskEvalDetail:
    """Stores detailed results for a single AC/DC task evaluation."""

    task_id: str
    instructions: str
    raw_output: str
    score: float


def _evaluate_acdc_task_sandbox_worker(
    args: Tuple[ACDCTask, str],
) -> Tuple[str, float, Optional[str], str]:
    """
    Worker function for multiprocessing pool to evaluate a single AC/DC task's response in a sandbox.

    Args:
        args: A tuple containing (acdc_task_instance, raw_llm_output).

    Returns:
        A tuple containing (task_id, score, instructions, raw_output).
        Instructions might be None if the task failed to load them initially.
        raw_output is passed through for result aggregation.
    """
    if args[0] is None:
        return None, 0.0, None, args[1]
    task, answer = args
    # The task instance already has the cfg needed for the sandbox call
    score = task.evaluate_response_sandboxed(answer)
    instructions = task.get_instructions()  # Retrieve cached instructions
    # Return task_id, score, instructions, and the original raw_output
    return task.task_id, score, instructions, answer


# def extract_task_instructions(task: ACDCTask, cfg: DictConfig):
def extract_task_instructions(task: ACDCTask):
    """Extract instructions from a task directory."""
    try:
        task_dir = task.task_dir
        instructions = task.get_instructions()

        # Turn all cfg.docker_sandbox keys and values into a dicts and lists
        # docker_sandbox_cfg = make_serializable(cfg.docker_sandbox)

        return {
            "task_dir": task_dir,
            # "docker_sandbox": docker_sandbox_cfg,
            "prompt": instructions,
        }
    except Exception as e:
        logging.error(f"Error extracting instructions from {task.task_id}: {e}")
        return None


def evaluate_acdc_tasks(
    prompts: List[str],
    task_dirs: List[str],
    proposed_answers: List[str],  # List of proposed answers corresponding to each task
    cfg: DictConfig,
    do_multiprocessing: bool = False,
) -> Tuple[Dict[str, float], float, List[ACDCTaskEvalDetail]]:
    """
    Evaluate a list of AC/DC tasks with proposed answers using parallel sandbox evaluation.

    Args:
        prompts: List of task prompts/instructions
        task_dirs: List of task directories
        proposed_answers: List of proposed answers corresponding to each task
        cfg: Hydra configuration object, expected to contain cfg.docker_sandbox settings.

    Returns:
        Tuple containing:
        - acdc_skill_vector: Dict[task_id, score] for AC/DC tasks
        - avg_acdc_quality: Average score across evaluated AC/DC tasks
        - acdc_eval_details: List of detailed evaluation results for AC/DC tasks
    """
    acdc_skill_vector: Dict[str, list[float]] = {}
    acdc_eval_details: List[ACDCTaskEvalDetail] = []
    acdc_quality_sum: float = 0.0
    acdc_task_count: int = 0

    # Prepare arguments for parallel evaluation
    sandbox_eval_args = []
    for i, (prompt, task_dir, answer) in enumerate(
        zip(prompts, task_dirs, proposed_answers)
    ):
        task_id = os.path.basename(task_dir)

        if not os.path.exists(task_dir):
            task = None
        else:
            # Create ACDCTask object
            task = ACDCTask(
                task_dir=task_dir,
                cfg=cfg,
            )

        sandbox_eval_args.append((task, answer))

    num_workers = cfg.acdc.get("num_sandbox_workers", multiprocessing.cpu_count())

    if do_multiprocessing:
        # Run parallel sandbox evaluation
        if sandbox_eval_args:
            logger.info(
                f"Starting parallel sandbox evaluation for {len(sandbox_eval_args)} AC/DC tasks using {num_workers} workers..."
            )

            try:
                with multiprocessing.Pool(processes=num_workers) as pool:
                    pool_results = pool.map(
                        _evaluate_acdc_task_sandbox_worker, sandbox_eval_args
                    )
                logger.info("Parallel sandbox evaluation complete.")

                # Aggregate results
                for task_id, score, instructions, answer in pool_results:
                    acdc_skill_vector[task_id] = score
                    acdc_quality_sum += score
                    acdc_task_count += 1

                    # Store evaluation details
                    acdc_eval_details.append(
                        ACDCTaskEvalDetail(
                            task_id=task_id,
                            instructions=instructions,
                            raw_output=answer,
                            score=score,
                        )
                    )
                    logger.debug(f"Processed ACDCTask {task_id}: score={score}")

            except Exception as e:
                logger.exception(f"Error during parallel sandbox evaluation: {e}")
                # Mark all remaining tasks as failed
                for prompt, task_dir, answer in zip(
                    prompts, task_dirs, proposed_answers
                ):
                    task_id = os.path.basename(task_dir)
                    if (
                        task_id not in acdc_skill_vector
                    ):  # Avoid overwriting existing results
                        acdc_skill_vector[task_id] = 0.0
                        acdc_eval_details.append(
                            ACDCTaskEvalDetail(
                                task_id=task_id,
                                instructions=prompt,
                                raw_output="<EVALUATION_FAILED>",
                                score=0.0,
                            )
                        )
                        acdc_task_count += 1
        else:
            logger.info("No valid AC/DC tasks to evaluate in sandbox.")

    else:
        # Run sequential evaluation
        for args in sandbox_eval_args:
            task_id, score, instructions, answer = _evaluate_acdc_task_sandbox_worker(
                args
            )
            acdc_skill_vector[task_id] = score
            acdc_quality_sum += score
            acdc_task_count += 1

            # Store evaluation details
            acdc_eval_details.append(
                ACDCTaskEvalDetail(
                    task_id=task_id,
                    instructions=instructions,
                    raw_output=answer,
                    score=score,
                )
            )
            logger.debug(f"Processed ACDCTask {task_id}: score={score}")
    # Calculate average quality
    avg_acdc_quality = (acdc_quality_sum / acdc_task_count) if acdc_task_count > 0 else None

    return acdc_skill_vector, avg_acdc_quality, acdc_eval_details


def generate_responses(
    model: LLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    sampling_params_dict: DictConfig,
) -> Tuple[List[List[str]], List[str]]:
    """Generate multiple responses for each prompt using vLLM."""
    if isinstance(tokenizer.eos_token_id, list):
        stop_token_ids = tokenizer.eos_token_id
    else:
        stop_token_ids = [tokenizer.eos_token_id]
    sampling_params = SamplingParams(
        temperature=sampling_params_dict.get("temperature", 0.0),
        max_tokens=sampling_params_dict.get("max_tokens", 512),
        top_p=sampling_params_dict.get("top_p", 1.0),
        stop_token_ids=stop_token_ids,
    )

    chat_prompts_tokenized = []
    for prompt in prompts:
        try:
            if prompt:
                # apply chat template to the prompt
                if sampling_params_dict.get("eval_cot", False):
                    messages = [
                        [
                            {
                                "role": "system",
                                "content": eval_cot_system_msg,
                            },
                            {"role": "user", "content": prompt},
                        ]
                    ]
                else:
                    messages = [
                        [
                            {
                                "role": "system",
                                "content": eval_zs_system_msg,
                            },
                            {"role": "user", "content": prompt},
                        ]
                    ]
                chat_formatted_message = TokensPrompt(
                    prompt_token_ids=tokenizer.apply_chat_template(
                        messages, tokenize=True
                    )[0],
                )
                chat_prompts_tokenized.append(chat_formatted_message)
        except Exception as e:
            logger.error(f"Error constructing or tokenizing prompt: {prompt}")
            logger.error(f"Error: {e}")
            chat_prompts_tokenized.append(
                TokensPrompt(
                    prompt_token_ids=[
                        sampling_params_dict.get("stop_token_ids", [128009])[-1]
                    ]
                )
            )

    outputs = model.generate(chat_prompts_tokenized, sampling_params)

    raw_outputs = []
    for output in outputs:
        raw_outputs.extend(o.text for o in output.outputs)

    # Extract the answer from the responses
    answers = []
    for raw_output in raw_outputs:
        answer = extract_answer_from_raw_output(raw_output)
        answers.append(answer)

    return raw_outputs, answers


def get_valid_tasks(tasks: List[ACDCTask], generated_tasks_dir: str) -> List[ACDCTask]:
    """Get valid tasks from the task pool."""
    valid_tasks = set()
    all_active_pool_files = glob.glob(
        os.path.join(generated_tasks_dir, "active_pool_*.json")
    )

    for active_pool_file in all_active_pool_files:
        with open(active_pool_file, "r") as f:
            active_pool = json.load(f)
        # remove the absolute path prefix to the output dir
        active_pool = [
            generated_tasks_dir + "/" + task_dir.split("/")[-1]
            for task_dir in active_pool
        ]
        valid_tasks.update(active_pool)

    # Remove tasks that are not in the valid tasks
    tasks_filtered = [task for task in tasks if task.task_dir in valid_tasks]

    return tasks_filtered


def remove_already_evaluated_models(
    model_paths: List[str], global_skill_vector_dir: str
) -> List[str]:
    """Remove models that have already been evaluated."""
    already_evaluated_models = glob.glob(
        os.path.join(global_skill_vector_dir, "*_skill_vector.json")
    )
    already_evaluated_models = [
        os.path.basename(model_path).split("_skill_vector.json")[0]
        for model_path in already_evaluated_models
    ]

    logger.info(
        f"Removing {len(already_evaluated_models)}/{len(model_paths)} already evaluated models from the list of models to evaluate."
    )

    model_paths = [
        model_path
        for model_path in model_paths
        if os.path.basename(model_path) not in already_evaluated_models
    ]
    return model_paths


@hydra.main(version_base=None, config_path="configs", config_name="ac_dc")
def main(cfg: DictConfig):
    output_dir = cfg.get("output_dir", None)
    all_dirs = setup_optimization_directories(cfg, output_dir)

    # Dir to save global skill vector per model
    global_skill_vector_dir = os.path.join(output_dir, "global_skill_vectors")
    os.makedirs(global_skill_vector_dir, exist_ok=True)

    # Initialize task pool
    logger.info("Initializing task pool...")
    task_pool = ACDCTaskPool(
        cfg, all_dirs["generated_tasks_dir"], all_dirs["vector_db_dir"]
    )

    logger.info("Loading existing task pool...")
    task_pool.load_existing_tasks()

    tasks: List[ACDCTask] = task_pool.get_tasks()
    tasks = get_valid_tasks(tasks, all_dirs["generated_tasks_dir"])
    logger.info(f"Found {len(tasks)} valid tasks in the task pool.")

    # Extract instructions for each task
    logger.info("Extracting task instructions...")
    task_instructions = []
    for task in tasks:
        instruction_data: list[dict] = extract_task_instructions(task)
        if instruction_data:
            task_instructions.append(instruction_data)

    prompts = [task_instruction["prompt"] for task_instruction in task_instructions]
    task_dirs = [task_instruction["task_dir"] for task_instruction in task_instructions]

    # Get model paths from config or command line override
    if cfg.get("model_paths_file", None):
        # Read model paths from a file (one path per line)
        logger.info(f"Reading model paths from {cfg.model_paths_file}")
        with open(cfg.model_paths_file, "r") as f:
            model_paths = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(model_paths)} model paths from file.")
    elif cfg.get("model_paths", None):
        # Get model paths directly from config (comma-separated or list)
        if isinstance(cfg.model_paths, str):
            model_paths = [p.strip() for p in cfg.model_paths.split(",") if p.strip()]
        else:
            model_paths = list(cfg.model_paths)
        logger.info(f"Using {len(model_paths)} model paths from config.")
    else:
        # Default behavior: scan the model directory
        model_paths = glob.glob(os.path.join(all_dirs["model_dir"], "*"))
        logger.info(f"Found {len(model_paths)} models in the model directory.")

    model_paths = remove_already_evaluated_models(model_paths, global_skill_vector_dir)

    if len(model_paths) == 0:
        logger.info("No models to evaluate. Exiting.")
        return

    logger.info(
        f"Evaluating {len(model_paths)} models on global task pool of {len(tasks)} tasks...\n"
    )

    # Load the base vLLM model
    model = LLM(
        model=cfg.base_model_path,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    # Set chat template
    # if cfg.chat_template == "llama3":
    #     tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    # elif cfg.chat_template == "qwen2_5":
    #     tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
    # elif cfg.chat_template == "qwen2":
    #     tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Loop over all models in archive and evaluate them on the task pool
    # Save each models new skill vector to a file
    # For every new model, update the vLLM model with the new weights

    for model_path in model_paths:
        try:
            model_name = os.path.basename(model_path)
            logger.info(f"Evaluating model {model_name} on global task pool...")

            # Load the model weights
            logger.info(f"Updating vLLM model with weights from {model_path}")
            state_dict = load_safetensors_state_dict(model_path)
            if "Qwen" in cfg.base_model_path:
                vllm_ready_state_dict = state_dict_hf_to_vllm_qwen(state_dict)
            elif "Llama-3" in cfg.base_model_path:
                vllm_ready_state_dict = state_dict_hf_to_vllm_llama(state_dict)
            else:
                raise ValueError
            update_vllm_weights(param=vllm_ready_state_dict, llm=model)

            # Get answers to instructions from the model
            logger.info("Generating responses from the model...")
            raw_outputs, answers = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                sampling_params_dict=cfg.vllm_pop,
            )

            # Evaluate the answers on the task pool
            acdc_skill_vector, _, _ = evaluate_acdc_tasks(
                prompts=prompts,
                task_dirs=task_dirs,
                proposed_answers=answers,
                cfg=cfg,
                do_multiprocessing=True,
            )

            logger.info(f"Evaluation done. Saving results...\n")
            # Save the global skill vector
            global_skill_vector_path = os.path.join(
                global_skill_vector_dir, f"{model_name}_skill_vector.json"
            )
            with open(global_skill_vector_path, "w") as f:
                json.dump(acdc_skill_vector, f)

            # Save the prompts, responses, and answers to a jsonl file
            details_log_path = os.path.join(
                global_skill_vector_dir, f"{model_name}_eval_details.jsonl"
            )
            log_prompts_and_responses(
                prompts=prompts,
                responses=raw_outputs,
                answers=answers,
                task_dirs=task_dirs,
                log_file_name=details_log_path,
                acdc_skill_vector=acdc_skill_vector,
            )
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            continue


if __name__ == "__main__":
    main()
