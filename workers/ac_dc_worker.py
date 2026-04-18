import time
import os
import sys
import json
from functools import partial

# import cma # Not used in this refactored version
import hydra
import logging
import numpy as np
import re
import torch
import multiprocessing  # Added import
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,  # type: ignore
    AutoTokenizer,  # type: ignore
    PreTrainedModel,  # type: ignore
    PreTrainedTokenizer,  # type: ignore
)
from typing import List, Tuple, Dict, Optional, Union, Any
from vllm import LLM, SamplingParams  # Added SamplingParams
import random
import backoff
from datatypes import (
    TaskMetric,
    ACDCMergeResult,
    ACDCTaskEvalDetail,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tasks.acdc_task import ACDCTask
from tasks.base import BaseTask
from utils.helpers import (
    load_hf_params_to_vllm,
    state_dict_hf_to_vllm_qwen,
    update_vllm_weights_general,
)
from tasks.task_gen_prompts import (
    GIBBERISH_PROMPT,
    GIBBERISH_PROMPT_REVISED,
    eval_cot_system_msg,
    eval_zs_system_msg,
)
from tasks.vllm_scientist import create_vllm_client_params, get_vllm_response


# --- Helper function for multiprocessing ---
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

    ### Look for "####" in the raw_output
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
    task, raw_output = args
    # The task instance already has the cfg needed for the sandbox call
    answer = extract_answer_from_raw_output(raw_output)
    score = task.evaluate_response_sandboxed(answer)
    instructions = task.get_instructions()  # Retrieve cached instructions
    # Return task_id, score, instructions, and the original raw_output
    return task.task_id, score, instructions, raw_output


# --- End Helper function ---


class ACDCWorker:

    def __init__(self, cfg: DictConfig):
        # ... (init remains largely the same, ensure logger is initialized)
        self.cfg = cfg
        self.logger = logging.getLogger("Worker")  # Basic logger first

        # Track vLLM request statistics
        self.vllm_request_count = 0
        self.vllm_success_count = 0
        self.vllm_failure_count = 0
        self.vllm_retry_count = 0

        # SVD task vector experts
        self.svd_expert_names = cfg.svd_expert_names

        # if "Qwen2.5" in cfg.base_model_path or "Qwen" in cfg.base_model_path:
        if "qwen" in cfg.base_model_path.lower():
            self.load_params_fn = partial(
                update_vllm_weights_general,
                hf_to_vllm_conversion_fn=state_dict_hf_to_vllm_qwen,
            )
        else:
            self.load_params_fn = load_hf_params_to_vllm

        gpu_memory_utilization = cfg.gpu_memory_utilization
        self.llm = LLM(
            cfg.base_model_path,
            max_model_len=2048,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            enforce_eager=False,
            # Increase max_num_seqs to allow batching for AC/DC tasks
            max_num_seqs=4,  # Allow batching up to pool size + buffer
            max_seq_len_to_capture=1024,
        )

        try:
            m = (
                self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model  # type: ignore
            )
            for _, param in m.named_parameters():
                param.requires_grad = False
        except AttributeError as e:
            # Handle case where vLLM internal structure has changed
            self.logger.warning(f"Could not access vLLM model parameters: {e}")

        # Initialize base model parameters
        self.hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path, torch_dtype=torch.bfloat16
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_path
        )
        self.base_params = self.hf_model.state_dict()

        # Set chat template
        if cfg.chat_template == "llama3":
            self.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        elif cfg.chat_template == "qwen2_5":
            self.tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
        elif cfg.chat_template == "qwen2":
            self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        elif cfg.chat_template == "deepseek_v1":
            self.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        elif cfg.chat_template == "qwen3":
            self.tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
        else:
            # Custom template as in config
            self.tokenizer.chat_template = cfg.chat_template

        # Initialize other components
        self.crossover = hydra.utils.instantiate(cfg.evo.crossover)
        self.mutator = hydra.utils.instantiate(cfg.evo.mutation)

        # Track current model state
        self.current_model_path = None
        self.current_model_params = None


    def load_model(self, model_path: str):
        """Load a model's parameters into the worker's LLM."""
        if model_path != self.current_model_path:
            self.logger.info(f"Loading model: {model_path}")
            self.logger.info(f"Old model: {self.current_model_path}")

            # If Ray is not initialized, load model directly
            self.current_model_params = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            ).state_dict()
            self.load_params_fn(self.current_model_params, self.llm)

            self.current_model_path = model_path
            self.logger.info(f"Loaded model: {model_path}")

    def merge_models_only(
        self,
        parent_paths: List[str],
        save_path: str,
        do_mutate: bool = True,
    ) -> Optional[str]:
        """Merge parent models and save without evaluation.

        Args:
            parent_paths: List of exactly 2 parent model paths.
            save_path: Where to save the merged model.
            do_mutate: Whether to apply mutation after crossover.

        Returns:
            str: Save path if successful, None if failed.
        """
        # Retry up to 3 times
        for _ in range(3):
            try:
                self.logger.info(
                    f"Merging models (no eval): parents={parent_paths}, save_path={save_path}, do_mutate={do_mutate}"
                )

                # Merge parents
                child_param = self.crossover.merge(
                    self.base_params,
                    parent_paths,
                    None,  # No CMA-ES parameters for now
                )
                self.logger.info(f"Crossover complete for {save_path}")

                # Only mutate if requested
                if do_mutate:
                    # Mutate child using a randomly selected task
                    random_task_name = np.random.choice(self.svd_expert_names)

                    # Given that the name of the task vector SVD in the file is "mbpp", if "humaneval" is in the config task name, change it to "mbpp"
                    if "humaneval" in random_task_name:
                        random_task_name = "mbpp"

                    start_time = time.time()
                    child_param = self.mutator.mutate(
                        child_param,
                        random_task_name,  # Use randomly selected task for mutation
                        None,  # No CMA-ES parameters for now
                    )
                    end_time = time.time()
                    self.logger.info(
                        f"Mutation complete using task {random_task_name} took {end_time - start_time:.2f} seconds"
                    )
                else:
                    self.logger.info("Skipping mutation as do_mutate=False")

                # Save the model without evaluation
                self.hf_model.load_state_dict(child_param)
                self.hf_model.save_pretrained(save_path, safe_serialize=True)
                # Save the tokenizer alongside the model
                self.tokenizer.save_pretrained(save_path)
                # Save the parent models mapping to joint JSONL file
                self._save_parent_mapping(save_path, parent_paths)
                self.logger.info(f"Saved merged model to {save_path}")

                return save_path

            except Exception as e:
                self.logger.error(
                    f"Retrying, failed to merge models at {save_path}: {e}"
                )

        self.logger.error(
            f"Failed to merge models at {save_path} after 3 retries"
        )
        return None

    def merge_models(
        self,
        parent_paths: List[str],
        save_path: str,
        task_info: Union[
            Dict[str, DictConfig], List[str]
        ],  # Add task_info parameter
        do_mutate: bool = True,
    ) -> Optional[ACDCMergeResult]:
        """Merge parent models, evaluate on specified tasks, and return results.

        Args:
            parent_paths: List of exactly 2 parent model paths.
            save_path: Where to save the merged model.
            task_info: Information to load tasks (dict for standard, list for AC/DC).
            do_mutate: Whether to apply mutation after crossover.

        Returns:
            ACDCMergeResult containing metrics and save path if successful.
        """
        # if an error occurs, retry 3 times
        for _ in range(3):
            try:
                self.logger.info(
                    f"Merging models: parents={parent_paths}, save_path={save_path}, do_mutate={do_mutate}"
                )

                # Merge parents
                child_param = self.crossover.merge(
                    self.base_params,
                    parent_paths,
                    None,  # No CMA-ES parameters for now
                )
                self.logger.info(f"Crossover complete for {save_path}")

                # Only mutate if requested
                if do_mutate:
                    # Mutate child using a randomly selected task
                    random_task_name = np.random.choice(self.svd_expert_names)

                    # Given that the name of the task vector SVD in the file is "mbpp", if "humaneval" is in the config task name, change it to "mbpp"
                    if "humaneval" in random_task_name:
                        random_task_name = "mbpp"

                    start_time = time.time()
                    child_param = self.mutator.mutate(
                        child_param,
                        random_task_name,  # Use randomly selected task for mutation
                        None,  # No CMA-ES parameters for now
                    )
                    end_time = time.time()
                    self.logger.info(
                        f"Mutation complete using task {random_task_name} took {end_time - start_time:.2f} seconds"
                    )
                else:
                    self.logger.info("Skipping mutation as do_mutate=False")

                # Evaluate the model before saving
                self.hf_model.load_state_dict(child_param)
                self.logger.info(f"HF state dict loaded")
                # self.load_params_fn(child_param, self.llm) # Done in _eval_model
                self.current_model_path = (
                    save_path  # Track path even before saving
                )
                self.logger.info(
                    f"HF params will be loaded to vllm for eval in {save_path}"
                )

                # Load tasks for this specific evaluation
                tasks = self._load_tasks_from_info(task_info, self.cfg)
                if not tasks:
                    self.logger.error(
                        f"No tasks loaded for merge evaluation of {save_path}. Skipping."
                    )
                    return None

                # Evaluate the merged and mutated model using the loaded tasks
                (
                    standard_metrics,
                    acdc_skill_vector,
                    avg_acdc_quality,
                    acdc_eval_details,
                    is_gibberish,
                ) = self._eval_model(
                    child_param, "train", tasks  # Pass child_param
                )
                self.logger.info(f"Evaluation complete for {save_path}")

                # Check if evaluation produced any results (might be None if all tasks failed/skipped)
                if standard_metrics is not None or acdc_skill_vector is not None:
                    # Save the model state dict
                    self.hf_model.load_state_dict(
                        child_param
                    )  # Ensure hf_model has the correct params before saving
                    self.hf_model.save_pretrained(
                        save_path, safe_serialize=True
                    )
                    # Save the tokenizer alongside the model
                    self.tokenizer.save_pretrained(save_path)
                    # Save the parent models mapping to joint JSONL file
                    self._save_parent_mapping(save_path, parent_paths)
                    self.logger.info(
                        f"Saved model and tokenizer to {save_path}"
                    )

                    # Return ACDCMergeResult with new structure, including eval details
                    return ACDCMergeResult(
                        save_path=save_path,
                        task_metrics=standard_metrics,
                        acdc_skill_vector=acdc_skill_vector,
                        avg_acdc_quality=avg_acdc_quality,
                        acdc_eval_details=acdc_eval_details,  # Pass collected details
                        is_gibberish=is_gibberish or False,
                    )
                else:
                    self.logger.error(
                        f"Model evaluation failed or yielded no results during merge for {save_path}"
                    )
                    return None

            except Exception as e:
                self.logger.error(
                    f"Retrying, failed to merge models at {save_path}: {e}"
                )
        self.logger.error(
            f"Failed to merge models at {save_path} after 3 retries"
        )
        return None

    def _eval_model(
        self, param: Dict, data_split: str, tasks: List[BaseTask]
    ) -> Tuple[
        Optional[Dict[str, TaskMetric]],
        Optional[Dict[str, float]],
        Optional[float],
        Optional[List[ACDCTaskEvalDetail]],
        Optional[bool],
    ]:
        """
        Internal method to evaluate model parameters on a given list of tasks.
        Handles standard tasks sequentially and AC/DC tasks with batched generation
        followed by parallel sandbox evaluation.

        Args:
            param: Model state dictionary.
            data_split: Data split to evaluate ('train', 'validation', 'all').
            tasks: List of BaseTask objects (standard or AC/DC) to evaluate.

        Returns:
            A tuple containing:
            - standard_metrics: Dict[task_name, TaskMetric] for non-AC/DC tasks.
            - acdc_skill_vector: Dict[task_id, score] for AC/DC tasks.
            - avg_acdc_quality: Average score across evaluated AC/DC tasks.
            - acdc_eval_details: List of detailed evaluation results for AC/DC tasks.
            - is_gibberish: Whether the model returns gibberish or not.
        """
        self.load_params_fn(
            param, self.llm
        )  # Load model params into vLLM instance

        standard_metrics: Dict[str, TaskMetric] = {}
        acdc_tasks: List[ACDCTask] = []
        other_tasks: List[BaseTask] = []

        # Separate tasks
        for task in tasks:
            if isinstance(task, ACDCTask):
                acdc_tasks.append(task)
            elif isinstance(task, BaseTask):  # Catch standard tasks
                other_tasks.append(task)
            else:
                self.logger.warning(
                    f"Task {getattr(task, 'task_name', 'Unknown')} has unrecognized type {type(task)}. Skipping evaluation."
                )

        # --- Evaluate Standard Tasks Sequentially ---
        for task in other_tasks:
            # Add retry logic for standard task evaluation as well
            @backoff.on_exception(
                backoff.expo,
                Exception,
                max_tries=self.cfg.evaluation.get("max_retries", 3),
                max_time=180,  # Maximum 3 minutes per task
                on_backoff=lambda details: self._handle_vllm_backoff(
                    details, f"Standard task {task.task_name}"
                ),
            )
            def evaluate_task_with_retry():
                self.vllm_request_count += 1
                start_time = time.time()
                try:
                    result = task.get_q_and_bc(self.llm, data_split=data_split)  # type: ignore
                    elapsed = time.time() - start_time
                    self.vllm_success_count += 1
                    self.logger.info(
                        f"Standard task evaluation succeeded in {elapsed:.2f}s. "
                        f"Queue stats: {self._get_vllm_stats()}"
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.vllm_failure_count += 1
                    self.logger.error(
                        f"Standard task evaluation failed after {elapsed:.2f}s: {e}. "
                        f"Queue stats: {self._get_vllm_stats()}"
                    )
                    raise

            try:
                task_metric = evaluate_task_with_retry()
                standard_metrics[task.task_name] = task_metric
                self.logger.debug(
                    f"Evaluated Standard Task {task.task_name}: quality={task_metric.quality}"
                )
            except Exception as e:
                task_identifier = getattr(task, "task_name", "Unknown")
                self.logger.exception(
                    f"Error evaluating standard task {task_identifier} after all retries: {e}"
                )
                # Add a placeholder metric indicating failure to prevent worker crash
                standard_metrics[task.task_name] = TaskMetric(
                    quality=0.0,
                    bc_ids=tuple([0] * task.bc_num_dims),
                    example_results={},
                )

        # --- Evaluate AC/DC Tasks (Batched Generation + Parallel Sandbox) ---
        acdc_skill_vector: Dict[str, float] = {}
        acdc_eval_details: List[ACDCTaskEvalDetail] = []
        acdc_quality_sum: float = 0.0
        acdc_task_count: int = 0

        if acdc_tasks:
            # 1. Prepare Prompts and Filter Tasks
            prompts_to_generate = []
            valid_acdc_tasks_for_gen = (
                []
            )  # Keep track of tasks corresponding to prompts
            for task in acdc_tasks:
                # check if task exists on disk
                if not os.path.exists(task.task_dir):
                    self.logger.warning(
                        f"Task {task.task_id} not found on disk. Skipping evaluation."
                    )
                    prompt = None
                else:
                    prompt = task.get_evaluation_prompt()
                self.logger.debug(f"Prompt before chat template: {prompt}")
                if prompt and isinstance(prompt, str):
                    # apply chat template to the prompt
                    if self.cfg.vllm_pop.get("eval_cot", False):
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
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False
                    )[0]
                    self.logger.debug(f"Prompt after chat template: {prompt}")
                    prompts_to_generate.append(prompt)
                    valid_acdc_tasks_for_gen.append(task)
                else:
                    self.logger.warning(
                        f"Skipping AC/DC task {task.task_id} due to missing prompt or task directory."
                    )
                    # Add placeholder failure result directly
                    acdc_skill_vector[task.task_id] = 0.0
                    acdc_eval_details.append(
                        ACDCTaskEvalDetail(
                            task_id=task.task_id,
                            instructions="<PROMPT FAILED>",
                            raw_output="<PROMPT FAILED>",
                            score=0.0,
                        )
                    )
                    # Don't increment acdc_quality_sum, but count as evaluated
                    acdc_task_count += 1

            if prompts_to_generate:
                # 2. Batched LLM Generation
                self.logger.info(
                    f"Generating responses for {len(prompts_to_generate)} AC/DC tasks..."
                )
                # Handle stop_token_ids - convert to list if provided, otherwise None
                stop_token_ids_cfg = self.cfg.vllm_pop.get("stop_token_ids", None)
                stop_token_ids = list(stop_token_ids_cfg) if stop_token_ids_cfg else None

                sampling_params = SamplingParams(
                    temperature=self.cfg.vllm_pop.get("temperature", 0.0),
                    top_p=self.cfg.vllm_pop.get("top_p", 1.0),
                    max_tokens=self.cfg.vllm_pop.get("max_tokens", 512),
                    stop_token_ids=stop_token_ids,
                )

                # Define retry logic for vLLM generation
                @backoff.on_exception(
                    backoff.expo,
                    Exception,  # Catch all exceptions for vLLM calls
                    max_tries=self.cfg.evaluation.get("max_retries", 3),
                    max_time=300,  # Maximum 5 minutes total
                    on_backoff=lambda details: self._handle_vllm_backoff(
                        details, "AC/DC task generation"
                    ),
                )
                def generate_with_retry():
                    self.vllm_request_count += 1
                    start_time = time.time()
                    try:
                        outputs = self.llm.generate(
                            prompts=prompts_to_generate,
                            sampling_params=sampling_params,
                        )
                        elapsed = time.time() - start_time
                        self.vllm_success_count += 1
                        self.logger.info(
                            f"vLLM generation succeeded in {elapsed:.2f}s. "
                            f"Queue stats: {self._get_vllm_stats()}"
                        )
                        return outputs
                    except Exception as e:
                        elapsed = time.time() - start_time
                        self.vllm_failure_count += 1
                        self.logger.error(
                            f"vLLM generation failed after {elapsed:.2f}s: {e}. "
                            f"Queue stats: {self._get_vllm_stats()}"
                        )
                        raise

                try:
                    outputs = generate_with_retry()
                    self.logger.info(
                        f"Generation complete for {len(outputs)} AC/DC tasks."
                    )

                    # 3. Prepare Data for Parallel Sandbox Evaluation
                    sandbox_eval_args = []
                    # Outputs list corresponds 1:1 with prompts_to_generate
                    for i, task in enumerate(valid_acdc_tasks_for_gen):
                        if i < len(outputs) and outputs[i].outputs:
                            raw_output = outputs[i].outputs[0].text.strip()
                        else:
                            self.logger.error(
                                f"LLM generation failed for AC/DC task {task.task_id}."
                            )
                            raw_output = "<GENERATION FAILED>"  # Mark failure
                        ### FOR DEBUGGING
                        answer = extract_answer_from_raw_output(raw_output)
                        self.logger.debug(f"Raw output: {raw_output}")
                        self.logger.debug(f"Answer: {answer}")
                        ### END FOR DEBUGGING
                        sandbox_eval_args.append((task, raw_output))

                    # 4. Parallel Sandbox Evaluation
                    # Use cpu_count unless specified otherwise in config (e.g., cfg.acdc.num_sandbox_workers)
                    num_workers = self.cfg.acdc.get(
                        "num_sandbox_workers", multiprocessing.cpu_count()
                    )
                    self.logger.info(
                        f"Starting parallel sandbox evaluation for {len(sandbox_eval_args)} AC/DC tasks using {num_workers} workers..."
                    )

                    pool_results = []
                    if sandbox_eval_args:  # Only start pool if there's work
                        # Ensure the pool uses a safe start method like 'spawn' if needed, especially on macOS/Windows
                        # Python 3.8+ defaults to 'spawn' on macOS. Linux defaults to 'fork'.
                        # 'fork' can be problematic with complex objects and threads (like vLLM might use internally).
                        # Explicitly setting 'spawn' might be safer, though potentially slower startup.
                        # ctx = multiprocessing.get_context("spawn")
                        # with ctx.Pool(processes=num_workers) as pool:
                        with multiprocessing.Pool(
                            processes=num_workers
                        ) as pool:  # Using default context for now
                            pool_results = pool.map(
                                _evaluate_acdc_task_sandbox_worker,
                                sandbox_eval_args,
                            )
                        self.logger.info(
                            "Parallel sandbox evaluation complete."
                        )
                    else:
                        self.logger.info(
                            "No valid AC/DC tasks with generated output to evaluate in sandbox."
                        )

                    # 5. Aggregate Results
                    for (
                        task_id,
                        score,
                        instructions,
                        raw_output,
                    ) in pool_results:
                        acdc_skill_vector[task_id] = score
                        acdc_quality_sum += score
                        acdc_task_count += 1
                        # Store details (instructions might be None if task failed init)
                        acdc_eval_details.append(
                            ACDCTaskEvalDetail(
                                task_id=task_id,
                                instructions=instructions
                                or "<NO_INSTRUCTIONS>",
                                raw_output=raw_output,
                                score=score,
                            )
                        )
                        self.logger.debug(
                            f"Processed ACDCTask {task_id}: score={score}"
                        )

                except Exception as gen_err:
                    self.logger.exception(
                        f"Error during vLLM generation for AC/DC tasks after all retries: {gen_err}"
                    )
                    # Mark all tasks in this batch as failed if generation crashes
                    # This prevents worker shutdown and allows continuation
                    for task in valid_acdc_tasks_for_gen:
                        if (
                            task.task_id not in acdc_skill_vector
                        ):  # Avoid overwriting prompt failures
                            acdc_skill_vector[task.task_id] = 0.0
                            acdc_eval_details.append(
                                ACDCTaskEvalDetail(
                                    task_id=task.task_id,
                                    instructions=task.get_instructions(),
                                    raw_output="<GENERATION FAILED AFTER RETRIES>",
                                    score=0.0,
                                )
                            )
                            acdc_task_count += 1  # Count as evaluated

        # Calculate average AC/DC quality
        avg_acdc_quality = (
            (acdc_quality_sum / acdc_task_count) if acdc_task_count > 0 else None
        )

        if self.cfg.dns.get("run_gibberish_check", False):
            is_gibberish = self._is_gibberish(acdc_eval_details)
        else:
            is_gibberish = False

        # Return None for components if they are empty, otherwise the dict/value
        return (
            standard_metrics if standard_metrics else None,
            acdc_skill_vector if acdc_skill_vector else None,
            avg_acdc_quality,
            acdc_eval_details if acdc_eval_details else None,
            is_gibberish,
        )

    def initialize_model_only(
        self,
        seed_model_paths: List[str],  # One or two paths
        save_path: str,
        seed: int,
        do_mutate: bool = True,
    ) -> Optional[str]:
        """Initialize a new model without evaluation (phase 1).

        Args:
            seed_model_paths: List of 1 or 2 paths to seed models.
            save_path: Where to save the new model.
            seed: Random seed for initialization.
            do_mutate: Whether to apply mutation.

        Returns:
            str: Save path if successful, None if failed.
        """
        try:
            self.logger.info(
                f"Initializing model (no eval): seed_model_paths={seed_model_paths}, save_path={save_path}"
            )

            if len(seed_model_paths) not in [1, 2]:
                raise ValueError(
                    f"Expected 1 or 2 seed models, got {len(seed_model_paths)}"
                )

            # Set seeds for initialization
            self.crossover.update_seed(seed)
            self.mutator.update_seed(seed)
            np.random.seed(seed)  # For task selection

            # If two models provided, do crossover
            if len(seed_model_paths) == 2:
                child_param = self.crossover.merge(
                    self.base_params,
                    [
                        seed_model_paths[0],
                        seed_model_paths[1],
                    ],  # Always exactly 2 parents
                    None,  # No CMA-ES parameters for initialization
                )
                self.logger.info("Crossover executed")
            elif len(seed_model_paths) == 1:
                child_param = AutoModelForCausalLM.from_pretrained(
                    seed_model_paths[0], torch_dtype=torch.bfloat16
                ).state_dict()
            else:
                raise ValueError(
                    f"Expected 1 or 2 seed models, got {len(seed_model_paths)}"
                )

            # Mutate parameters if requested
            if do_mutate:
                random_task_name = np.random.choice(self.svd_expert_names)

                # Given that the name of the task vector SVD in the file is "mbpp", if "humaneval" is in the config task name, change it to "mbpp"
                if "humaneval" in random_task_name:
                    random_task_name = "mbpp"

                time_start = time.time()
                child_param = self.mutator.mutate(
                    child_param,
                    random_task_name,  # Use randomly selected task for mutation
                    None,  # No CMA-ES parameters for initialization
                )
                time_end = time.time()
                self.logger.info(
                    f"Mutation complete using task {random_task_name} in {time_end - time_start:.2f}s"
                )

            # Save the model without evaluation
            self.hf_model.load_state_dict(child_param)
            self.hf_model.save_pretrained(save_path, safe_serialize=True)
            # Save the tokenizer alongside the model
            self.tokenizer.save_pretrained(save_path)
            # Save the parent models mapping to joint JSONL file
            self._save_parent_mapping(save_path, seed_model_paths)
            self.logger.info(f"Saved initialized model to {save_path}")

            return save_path

        except Exception as e:
            self.logger.error(f"Failed to initialize model at {save_path}: {e}")
            return None

    def _is_gibberish(self, acdc_eval_details: List[ACDCTaskEvalDetail]) -> bool:
        """Determines if the model returns gibberish based on AC/DC task eval details."""
        self.logger.info("Checking if model returns gibberish")

        # 1. Get all raw outputs with the corresponding score
        raw_outputs = [detail.raw_output for detail in acdc_eval_details]
        instructions = [detail.instructions for detail in acdc_eval_details]
        scores = [detail.score for detail in acdc_eval_details]

        # # 2. Get the top 3 outputs based on the score
        # top_3_outputs = sorted(
        #     zip(scores, raw_outputs, instructions),
        #     key=lambda x: x[0],
        #     reverse=True,
        # )[:3]
        # 2. Get 3 random outputs
        tuples = list(zip(scores, raw_outputs, instructions))
        random.shuffle(tuples)
        top_3_outputs = tuples[:3]
        top_3_outputs_str = "\n".join(
            [
                f"[RESPONSE {i+1}]\n{output}"
                for i, (_, output, _) in enumerate(top_3_outputs)
            ]
        )
        instructions = "\n".join(
            [
                f"[TASK {i+1}]\n{instruction}"
                for i, (_, _, instruction) in enumerate(top_3_outputs)
            ]
        )

        # 3. Format them into the prompt template
        prompt = GIBBERISH_PROMPT.format(
            outputs=top_3_outputs_str, instructions=instructions
        )
        # prompt = GIBBERISH_PROMPT_REVISED.format(
        #     outputs=top_3_outputs_str, instructions=instructions
        # )
        # log the prompt
        self.logger.info(f"Gibberish Check Prompt:\n{prompt}")

        # 4. create client
        try:
            vllm_client_params = create_vllm_client_params(self.cfg)
            if vllm_client_params is None:
                self.logger.error("Failed to create vLLM client parameters")
                return False

            # Apply retry logic with exponential backoff
            @backoff.on_exception(
                backoff.expo,
                Exception,  # Catch all exceptions
                max_tries=vllm_client_params.get("max_retries", 3),
                max_time=180,  # Maximum 3 minutes total
                on_backoff=lambda details: self._handle_vllm_backoff(
                    details, "Gibberish check"
                ),
            )
            def get_vllm_response_with_retry():
                self.vllm_request_count += 1
                start_time = time.time()
                try:
                    result = get_vllm_response(
                        prompt=prompt,
                        system_message="You are a helpful assistant",
                        base_url=vllm_client_params["base_url"],
                        model_name=vllm_client_params["model_name"],
                        temperature=vllm_client_params["temperature"],
                        max_tokens=vllm_client_params["max_tokens"],
                        top_p=vllm_client_params["top_p"],
                        timeout=vllm_client_params["timeout"],
                    )
                    elapsed = time.time() - start_time
                    self.vllm_success_count += 1
                    self.logger.info(
                        f"Gibberish check succeeded in {elapsed:.2f}s. "
                        f"Queue stats: {self._get_vllm_stats()}"
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.vllm_failure_count += 1
                    self.logger.error(
                        f"Gibberish check failed after {elapsed:.2f}s: {e}. "
                        f"Queue stats: {self._get_vllm_stats()}"
                    )
                    raise

            response, _ = get_vllm_response_with_retry()
            # log the response
            self.logger.info(f"Gibberish Check Response:\n{response}")
        except Exception as e:
            self.logger.exception(
                f"Error during vLLM generation for gibberish test after all retries: {e}"
            )
            # Return False instead of crashing the worker
            return False

        # 5. parse response
        if "Answer: Yes" in response:
            self.logger.info("Model returns gibberish")
            return True
        else:
            self.logger.info("Model does not return gibberish")
            return False

    def eval_model_only(
        self,
        model_path: str,
        task_info: Union[Dict[str, DictConfig], List[str]],
        data_split: str = "train",
        task_name: Optional[str] = None,
    ) -> Optional[ACDCMergeResult]:
        """Evaluate a pre-saved model without any merging.

        Args:
            model_path: Path to the saved model to evaluate.
            task_info: Task information for evaluation.
            data_split: Data split to evaluate on.
            task_name: Optional filter for specific task.

        Returns:
            ACDCMergeResult with evaluation results.
        """
        try:
            # Load model parameters from path
            model_param = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            ).state_dict()
            self.logger.info(f"Loaded model params from: {model_path}")

            # Load tasks based on task_info
            tasks = self._load_tasks_from_info(task_info, self.cfg)
            if not tasks:
                self.logger.error(
                    f"No tasks loaded for eval_model_only call for {model_path}. Returning empty result."
                )
                return None

            # Filter tasks if task_name is provided
            if task_name:
                tasks_to_evaluate = [
                    t
                    for t in tasks
                    if getattr(t, "task_name", None) == task_name
                    or getattr(t, "task_id", None) == task_name
                ]
                if not tasks_to_evaluate:
                    self.logger.warning(
                        f"Task '{task_name}' not found in loaded tasks for {model_path}. Evaluating all loaded tasks instead."
                    )
                    tasks_to_evaluate = tasks
            else:
                tasks_to_evaluate = tasks

            # Call the internal evaluation method with the loaded parameters
            (
                standard_metrics,
                acdc_skill_vector,
                avg_acdc_quality,
                acdc_eval_details,
                is_gibberish,
            ) = self._eval_model(model_param, data_split, tasks_to_evaluate)
            self.logger.info(f"Eval finished for {model_path}")

            # Return ACDCMergeResult with new structure, including details
            return ACDCMergeResult(
                save_path=model_path,
                task_metrics=standard_metrics,
                acdc_skill_vector=acdc_skill_vector,
                avg_acdc_quality=avg_acdc_quality,
                acdc_eval_details=acdc_eval_details,
                is_gibberish=is_gibberish or False,
            )
        except Exception as e:
            self.logger.error(f"Failed to evaluate model {model_path}: {e}")
            return None

    def eval_model(
        self,
        model_path: str,
        save_path: str,
        data_split: str,
        task_info: Union[Dict[str, DictConfig], List[str]],  # Add task_info
        task_name: Optional[str] = None,  # Keep optional task_name filter
    ) -> ACDCMergeResult:
        # Load model parameters from path
        model_param = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).state_dict()
        # No need to load into self.hf_model here unless other methods rely on it
        # self.hf_model.load_state_dict(model_param)
        self.logger.info(f"Loaded model params from: {model_path}")

        # Load tasks based on task_info
        tasks = self._load_tasks_from_info(task_info, self.cfg)
        if not tasks:
            self.logger.error(
                f"No tasks loaded for eval_model call for {model_path}. Returning empty result."
            )
            return ACDCMergeResult(save_path=save_path)

        # Filter tasks if task_name is provided
        if task_name:
            tasks_to_evaluate = [
                t
                for t in tasks
                if getattr(t, "task_name", None) == task_name
                or getattr(t, "task_id", None) == task_name
            ]
            if not tasks_to_evaluate:
                self.logger.warning(
                    f"Task '{task_name}' not found in loaded tasks for {model_path}. Evaluating all loaded tasks instead."
                )
                tasks_to_evaluate = (
                    tasks  # Fallback to all if specific task not found
                )
        else:
            tasks_to_evaluate = tasks

        # Call the internal evaluation method with the loaded parameters
        (
            standard_metrics,
            acdc_skill_vector,
            avg_acdc_quality,
            acdc_eval_details,
            is_gibberish,
        ) = self._eval_model(model_param, data_split, tasks_to_evaluate)
        self.logger.info(f"Eval finished for {model_path}")

        # Return ACDCMergeResult with new structure, including details
        return ACDCMergeResult(
            save_path=save_path,
            task_metrics=standard_metrics,
            acdc_skill_vector=acdc_skill_vector,
            avg_acdc_quality=avg_acdc_quality,
            acdc_eval_details=acdc_eval_details,  # Pass details
            is_gibberish=is_gibberish or False,
        )

    # ... (rest of the class methods like _get_input_info, initialize_model, setup_worker, _load_tasks_from_info remain the same)
    def _get_input_info(
        self, task: BaseTask, tasks: List[BaseTask]
    ) -> Tuple[int, List]:
        """Get input dimensionality info for CMA-ES."""
        target_task_name = task.task_name
        input_size = 0
        input_grid_sizes = []
        for t in tasks:
            if t.task_name != target_task_name:
                input_size += t.bc_num_dims
                input_grid_sizes.extend(t.bc_grid_sizes)
        return input_size * 2, input_grid_sizes

    def initialize_model(
        self,
        seed_model_paths: List[str],  # One or two paths
        save_path: str,
        seed: int,
        task_info: Union[Dict[str, DictConfig], List[str]],  # Add task_info
        do_mutate: bool = True,
    ) -> Optional[ACDCMergeResult]:
        """Initialize a new model by either mutating one seed model or crossing over two seed models.

        Args:
            seed_model_paths: List of 1 or 2 paths to seed models. If 1, only mutation is applied.
                            If 2, crossover is performed before mutation.
            save_path: Where to save the new model
            seed: Random seed for initialization

        Returns:
            ACDCMergeResult containing QD info, task metrics and save path if successful
        """
        try:
            self.logger.info(f"seed_model_paths={seed_model_paths}")
            self.logger.info(f"save_path={save_path}")

            if len(seed_model_paths) not in [1, 2]:
                raise ValueError(
                    f"Expected 1 or 2 seed models, got {len(seed_model_paths)}"
                )

            # Set seeds for initialization
            self.crossover.update_seed(seed)
            self.mutator.update_seed(seed)
            np.random.seed(seed)  # For task selection

            # If two models provided, do crossover
            if len(seed_model_paths) == 2:
                child_param = self.crossover.merge(
                    self.base_params,
                    [
                        seed_model_paths[0],
                        seed_model_paths[1],
                    ],  # Always exactly 2 parents
                    None,  # No CMA-ES parameters for initialization
                )
                self.logger.info("Crossover executed")
            elif len(seed_model_paths) == 1:
                child_param = AutoModelForCausalLM.from_pretrained(
                    seed_model_paths[0], torch_dtype=torch.bfloat16
                ).state_dict()
            else:
                raise ValueError(
                    f"Expected 1 or 2 seed models, got {len(seed_model_paths)}"
                )

            # Always mutate parameters using a randomly selected task
            random_task_name = np.random.choice(self.svd_expert_names)

            if do_mutate:
                start_time = time.time()
                child_param = self.mutator.mutate(
                    child_param,
                    random_task_name,  # Use randomly selected task for mutation
                    None,  # No CMA-ES parameters for initialization
                )
                end_time = time.time()
                self.logger.info(
                    f"Mutation complete using task {random_task_name} took {end_time - start_time:.2f} seconds"
                )

            # Evaluate the model before saving
            # self.hf_model.load_state_dict(child_param) # Not strictly needed if _eval_model loads params
            # self.load_params_fn(child_param, self.llm) # Done inside _eval_model
            self.current_model_path = save_path  # Track path

            # Load tasks for this specific evaluation
            tasks = self._load_tasks_from_info(task_info, self.cfg)
            if not tasks:
                self.logger.error(
                    f"No tasks loaded for initialization evaluation of {save_path}. Skipping."
                )
                return None

            # Capture all return values from _eval_model
            (
                standard_metrics,
                acdc_skill_vector,
                avg_acdc_quality,
                acdc_eval_details,
                is_gibberish,
            ) = self._eval_model(child_param, "train", tasks)
            self.logger.info(
                f"Evaluation complete for initialized model {save_path}"
            )

            # Only save if evaluation was successful
            if standard_metrics is not None or acdc_skill_vector is not None:
                self.hf_model.load_state_dict(
                    child_param
                )  # Ensure hf_model has the correct params FOR SAVING
                self.hf_model.save_pretrained(save_path, safe_serialize=True)
                # Save the tokenizer alongside the model
                self.tokenizer.save_pretrained(save_path)
                # Save the parent models mapping to joint JSONL file
                self._save_parent_mapping(save_path, seed_model_paths)
                self.logger.info(f"Saved initialized model to {save_path}")

                # Return ACDCMergeResult with new structure, including details
                return ACDCMergeResult(
                    save_path=save_path,
                    task_metrics=standard_metrics,
                    acdc_skill_vector=acdc_skill_vector,
                    avg_acdc_quality=avg_acdc_quality,
                    acdc_eval_details=acdc_eval_details,  # Pass details
                    is_gibberish=is_gibberish or False,
                )
            else:
                self.logger.error(
                    "Model evaluation failed or yielded no results during initialization"
                )
                return None  # Return None if evaluation failed

        except Exception as e:
            self.logger.exception(
                f"Failed to initialize model at {save_path}: {e}"
            )  # Use logger.exception to include traceback
            return None  # Return None on exception

    def initialize_random_seeds(self, seed: int) -> bool:
        """Initialize random seeds for worker components without loading a model.

        Args:
            seed: Random seed for initialization

        Returns:
            bool: True if successful
        """
        try:
            # Set seeds for initialization
            self.crossover.update_seed(seed)
            self.mutator.update_seed(seed)
            np.random.seed(seed)  # For task selection
            torch.manual_seed(seed)
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize random seeds: {e}")
            return False

    def setup_worker(self, output_dir: str) -> bool:
        """Set up the worker with the provided output directory.
        Should be called after the output directory is determined by the main process.

        Args:
            output_dir: Base output directory path determined by main process
        """
        # Ensure logger is fully setup if not done in init
        if not self.logger.handlers:  # Check if handlers were already added
            log_dir = os.path.join(output_dir, "worker_logs")
            os.makedirs(log_dir, exist_ok=True)

            # Create a unique log file for each worker
            worker_rank = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            logger_id = worker_rank
            log_file = os.path.join(log_dir, f"worker_{logger_id}.log")

            # Setup file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)  # Ensure logger level is set

            self.logger.info(f"Worker {worker_rank} logging to {log_file}")
        else:
            self.logger.info("Logger already configured.")

        return True  # Keep return for compatibility if needed

    def _load_tasks_from_info(
        self,
        task_info: Union[Dict[str, DictConfig], List[str], None],
        cfg: DictConfig,
    ) -> List[Any]:
        """Load task objects based on the provided information."""
        tasks: List[Any] = []
        if isinstance(task_info, dict):
            # Standard tasks: Info is a dictionary of configurations
            self.logger.debug("Loading standard tasks from config dict.")
            for task_name, config in task_info.items():
                try:
                    tasks.append(hydra.utils.instantiate(config))
                except Exception as e:
                    self.logger.error(
                        f"Failed to instantiate standard task {task_name}: {e}"
                    )
        elif isinstance(task_info, list):
            # AC/DC tasks: Info is a list of task directories
            self.logger.debug(
                f"Loading {len(task_info)} AC/DC tasks from directory list."
            )
            for task_dir in task_info:
                if not os.path.exists(task_dir):
                    raise FileNotFoundError(f"Task directory {task_dir} does not exist.")
                try:
                    # Pass cfg to ACDCTask constructor if needed (e.g., for sandbox config)
                    tasks.append(ACDCTask(task_dir=task_dir, cfg=cfg))
                except FileNotFoundError:
                    self.logger.error(
                        f"AC/DC task directory not found: {task_dir}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to instantiate ACDCTask from {task_dir}: {e}"
                    )
        elif task_info is None:
            self.logger.warning(
                "Received None for task_info, no tasks will be loaded."
            )
        else:
            self.logger.error(f"Unrecognized task_info type: {type(task_info)}")

        self.logger.info(f"Loaded {len(tasks)} tasks for current evaluation.")
        return tasks

    def _handle_vllm_backoff(self, details: Any, context: str) -> None:
        """Handle backoff logging and statistics tracking."""
        self.vllm_retry_count += 1
        wait_time = details.get("wait", 0)
        tries = details.get("tries", 0)
        exception = details.get("exception", "Unknown")

        self.logger.warning(
            f"vLLM request for {context} failed, backing off {wait_time:.1f}s "
            f"after {tries} tries. Exception: {exception}. "
            f"Queue stats: {self._get_vllm_stats()}"
        )

    def _save_parent_mapping(
        self, model_path: str, parent_paths: List[str]
    ) -> None:
        """Save parent mapping in a new directory for this tracking purpose."""
        try:
            parent_parent_dir = os.path.dirname(os.path.dirname(model_path))

            # Create a new directory for this tracking purpose
            parent_tracking_dir = os.path.join(
                parent_parent_dir, "parent_models_mapping"
            )
            os.makedirs(parent_tracking_dir, exist_ok=True)

            # Save the parent models mapping to joint JSONL file
            model_name = os.path.basename(model_path)
            joint_file = os.path.join(parent_tracking_dir, f"{model_name}.json")

            with open(joint_file, "w") as f:
                json.dump(parent_paths, f)

        except Exception as e:
            self.logger.error(f"Failed to save parent mapping: {e}")

    def _get_vllm_stats(self) -> str:
        """Get formatted vLLM request statistics."""
        success_rate = (
            self.vllm_success_count / self.vllm_request_count * 100
            if self.vllm_request_count > 0
            else 0
        )
        return (
            f"Total: {self.vllm_request_count}, "
            f"Success: {self.vllm_success_count} ({success_rate:.1f}%), "
            f"Failed: {self.vllm_failure_count}, "
            f"Retries: {self.vllm_retry_count}"
        )
