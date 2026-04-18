import json
import os
import logging
import random
import pathlib
import re
import shutil
import multiprocessing
import concurrent.futures

import tasks.task_gen_prompts  # Import the prompts module

# from functools import partial # Unused import
from typing import Optional, Dict, List, Tuple, Any
import tempfile

from omegaconf import DictConfig

# AC/DC imports (assuming AC/DC is importable or paths adjusted)
# Uncomment the necessary import
from tasks.tasks_utils import (
    save_task_to_disk,
    load_task_family,
    # maybe_create_task_embedding,
    # update_task_metadata,
)

# Local imports
# Import the new function name and the main response function
from tasks.vllm_scientist import create_vllm_client_params, get_vllm_response
from tasks.docker_sandbox import run_task_in_sandbox  # Import sandbox runner
from tasks.acdc_task import ACDCTask
from datatypes import ACDCSolution, ACDCArchiveData  # Import ACDCSolution
from tasks.task_gen_prompts import (  # Keep existing imports for adaptation prompts
    initial_task_gen_prompt_completely_novel,
    initial_task_gen_prompt_adapt_similar,
    task_creation_reflexion_prompt,
    # task_creation_system_msg, # No longer directly imported, loaded dynamically
    make_task_harder_prompt,
    make_task_easier_prompt,
    make_task_novel_prompt,
    make_task_novel_but_similar_prompt,
    regenerate_more_novel_tasks_prompt,
    interestingly_new_system_msg,
    interestingly_new_prompt,
)
from tasks.simple_vectordb import SimpleVectorDB


logger = logging.getLogger(__name__)


def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


class ACDCTaskPool:
    """
    Manages a pool of AC/DC-generated tasks, including initial generation.
    """

    def __init__(
        self, cfg: DictConfig, generated_tasks_dir: str, vector_db_dir: str
    ):
        """
        Initialize the task pool manager.

        Args:
            cfg: Configuration object containing AC/DC settings.
            generated_tasks_dir: The absolute path to store generated tasks.
            vector_db_dir: The absolute path to the vector database.
        """
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration checks
        # Configuration checks (removed generated_tasks_dir check)
        if (
            not cfg.acdc.get("seed_tasks_dir")
            or not cfg.acdc.get("initial_pool_size")
            or not cfg.acdc.get("scientist_model")
        ):
            raise ValueError(
                "Missing required AC/DC configuration in cfg.acdc (seed_tasks_dir, initial_pool_size, scientist_model)"
            )

        self.seed_tasks_dir = cfg.acdc.seed_tasks_dir
        self.generated_tasks_dir = generated_tasks_dir  # Use passed argument
        self.initial_pool_size = cfg.acdc.initial_pool_size
        self.scientist_model_id = cfg.acdc.scientist_model

        self.tasks: List[str] = []
        self.tasks_pass_rates: Dict[str, float] = {}
        self.random_instance = random.Random(cfg.seed if cfg.seed > 0 else 42)
        self.task_counter = 0  # Global counter for task naming

        # Initialize conditional parent replacement feature
        self.enable_conditional_parent_replacement = (
            self.cfg.task_generation.get(
                "experimental_conditional_parent_replacement", False
            )
        )
        self.pending_limbo_parents: Dict[int, str] = (
            {}
        )  # Maps child_task_number_for_generation_attempt -> original_parent_task_path
        self.active_limbo_map: Dict[str, str] = (
            {}
        )  # Maps successful_child_task_path -> original_parent_task_path

        # Initialize vLLM scientist client
        self.vllm_client_params: Optional[Dict[str, Any]] = None

        # Load the configured task creation prompt
        self.task_creation_prompt_name = cfg.acdc.get(
            "task_creation_prompt_name",
            "task_creation_system_msg",  # Default to original if not set
        )
        try:
            self.task_creation_system_prompt_str = getattr(
                tasks.task_gen_prompts, self.task_creation_prompt_name
            )
            self.logger.info(
                f"Using task creation prompt: {self.task_creation_prompt_name}"
            )
        except AttributeError:
            self.logger.error(
                f"Configured task creation prompt '{self.task_creation_prompt_name}' not found in dns.task_gen_prompts. "
                f"Falling back to default 'task_creation_system_msg'."
            )
            # Fallback to the original prompt if the configured one isn't found
            self.task_creation_system_prompt_str = (
                tasks.task_gen_prompts.task_creation_system_msg
            )

        # Only vLLM scientists are supported
        if not self.scientist_model_id.startswith("vllm/"):
            raise ValueError(
                f"Only vLLM scientists are supported. scientist_model must start with 'vllm/', got: {self.scientist_model_id}"
            )

        self.vllm_client_params = create_vllm_client_params(cfg)
        if self.vllm_client_params:
            self.logger.info(
                f"Using vLLM scientist: {self.vllm_client_params['model_name']} at {self.vllm_client_params['base_url']}"
            )
        else:
            if cfg.acdc.get("vllm_enabled", False):
                raise ValueError(
                    "vLLM is enabled but failed to configure vLLM client parameters."
                )
            else:
                raise ValueError(
                    "vLLM scientist specified but vllm_enabled is false."
                )

        # Create generated tasks directory
        pathlib.Path(self.generated_tasks_dir).mkdir(
            parents=True, exist_ok=True
        )

        ### Vector DBs
        self.vector_db_dir_active = None
        self.vector_db_active = None
        self.vector_db_dir_historical = None
        self.vector_db_historical = None
        if self.cfg.task_generation.get("do_similarity_search", False):
            ## Active vector DB
            self.vector_db_dir_active = vector_db_dir + "_active"
            pathlib.Path(self.vector_db_dir_active).mkdir(
                parents=True, exist_ok=True
            )

            self.vector_db_active = SimpleVectorDB(
                embedding_model_name=self.cfg.task_generation.vector_db.get(
                    "embedding_model_name", "all-MiniLM-L6-v2"
                ),
                embedding_vllm_url=self.cfg.task_generation.vector_db.get(
                    "embedding_vllm_url", "http://localhost:8010/v1"
                ),
                storage_path=self.vector_db_dir_active,
                max_seq_length=self.cfg.task_generation.vector_db.get(
                    "max_seq_length", 1024
                ),
                task_representation_vector_db=self.cfg.task_generation.vector_db.get(
                    "task_representation_vector_db", "metadata"
                ),
            )

            self.logger.info(
                f"Active vector DB initialized in {self.vector_db_dir_active}."
            )

            ## Historical vector DB
            self.vector_db_dir_historical = vector_db_dir + "_historical"
            pathlib.Path(self.vector_db_dir_historical).mkdir(
                parents=True, exist_ok=True
            )

            self.vector_db_historical = SimpleVectorDB(
                embedding_model_name=self.cfg.task_generation.vector_db.get(
                    "embedding_model_name", "all-MiniLM-L6-v2"
                ),
                embedding_vllm_url=self.cfg.task_generation.vector_db.get(
                    "embedding_vllm_url", "http://localhost:8010/v1"
                ),
                storage_path=self.vector_db_dir_historical,
                max_seq_length=self.cfg.task_generation.vector_db.get(
                    "max_seq_length", 1024
                ),
                task_representation_vector_db=self.cfg.task_generation.vector_db.get(
                    "task_representation_vector_db", "metadata"
                ),
            )

            self.logger.info(
                f"Historical vector DB initialized in {self.vector_db_dir_historical}."
            )

        self.logger.info(
            f"AC/DC Task Pool initialized. Generated tasks will be stored in: {self.generated_tasks_dir}"
        )

    def _get_scientist_response(
        self,
        prompt: str,
        system_msg: str,
        msg_history: Optional[List] = None,
    ) -> Tuple[Optional[str], List]:
        """Helper to call the appropriate scientist LLM."""
        if msg_history is not None:
            new_msg_w_history = msg_history + [
                {"role": "user", "content": prompt}
            ]
        else:
            new_msg_w_history = [{"role": "user", "content": prompt}]

        # Use vLLM scientist (only supported option)
        try:
            # Pass parameters from the stored dictionary
            assistant_message, full_history = get_vllm_response(
                prompt=new_msg_w_history,
                system_message=system_msg,
                base_url=self.vllm_client_params["base_url"],
                model_name=self.vllm_client_params["model_name"],
                temperature=self.vllm_client_params["temperature"],
                max_tokens=self.vllm_client_params["max_tokens"],
                top_p=self.vllm_client_params["top_p"],
                timeout=self.vllm_client_params["timeout"],
                # api_key is handled internally by get_vllm_response
                # max_retries is handled by backoff decorator if applied there, or caller needs to handle
            )

            user_asssitant_history = new_msg_w_history + [
                {"role": "assistant", "content": assistant_message}
            ]

            return assistant_message, user_asssitant_history
        except Exception as e:
            self.logger.error(
                f"Error calling vLLM scientist via OpenAI client: {e}"
            )
            return None, msg_history if msg_history is not None else []

    def _validate_generated_task(self, response: Dict) -> bool:
        """Basic validation for the generated task JSON."""
        required_fields = [
            "name_of_task",
            "description_of_task",
            "capability_being_measured",
            "estimated_human_difficulty",
            "task_family",
            "example_instruction",
        ]
        try:
            for field in required_fields:
                if field not in response:
                    self.logger.warning(
                        f"Generated task missing required field: {field}"
                    )
                    return False
        except Exception as e:
            self.logger.error(f"Error while validating task: {e}")
            return False
        return True

    def _eval_task(self, response: Dict) -> Tuple[str, bool]:
        """Evaluate the task using the LLM."""
        with tempfile.TemporaryDirectory() as anon_dir:
            # Ensure all required fields exist.
            required_fields = [
                "name_of_task",
                "description_of_task",
                "capability_being_measured",
                "estimated_human_difficulty",
                "task_family",
                "example_instruction",
                "done",
            ]
            for field in required_fields:
                try:
                    if field not in response:
                        return f"Field {field} missing from response.", False
                except Exception as e:
                    return f"Error while validating task: {e}", False

            # Ensure estimated human difficulty is valid
            if str(response["estimated_human_difficulty"]) not in [
                "1",
                "2",
                "3",
                "4",
                "5",
            ]:
                return (
                    f"Invalid estimated_human_difficulty: {response['estimated_human_difficulty']}, must be 1-5.",
                    False,
                )

            # Save to disk first.
            try:
                save_task_to_disk(anon_dir, response)
            except Exception as e:
                return f"Error while saving task to disk: {e}", False

            # Load and run task.
            # If the task fails, we return the exception as the answer.
            _, score, answer, _ = self._sandbox_validate_task(
                anon_dir, generate_answer=True
            )
            if score is None:
                return (
                    f"Sandbox validation failed for {anon_dir} with the exception: {answer}. Removing directory.",
                    False,
                )
            else:
                return (
                    f"Task run successfully. Agent achieved score: {score:.1f}/1.0.\nThe agent's raw submission was: '{answer}'",
                    True,
                )

    def _reflect_on_task(
        self,
        response: Dict,
        task_number: int,
        message_history: Optional[List] = None,
    ) -> Tuple[Optional[Dict], List]:
        """Reflect on the task to ensure it makes sense."""
        success = False
        if response is None:
            return None, []
        try:
            for j in range(self.cfg.task_generation.get("max_reflections", 0)):
                eval_response, _ = self._eval_task(response)

                self.logger.info(f"Eval response: {eval_response}")

                response_str, message_history = self._get_scientist_response(
                    task_creation_reflexion_prompt.format(
                        current_round=j + 1,  # used to be j+2
                        num_rounds=self.cfg.task_generation.get(
                            "max_reflections", 0
                        ),
                        eval_response=eval_response,
                    ),
                    self.task_creation_system_prompt_str.format(
                        num_rounds=self.cfg.task_generation.get(
                            "max_reflections", 0
                        )
                    ),
                    message_history,
                )
                parsed_response = extract_json_between_markers(response_str)
                if parsed_response is None:
                    self.logger.warning(
                        f"Failed to parse JSON from response: {response_str}"
                    )
                    continue
                response = parsed_response
                if response.get("done", "False").lower() == "true":
                    # used to be j+2
                    print(
                        f"Task {task_number}: Task generation converged after {j+1} self-reflection rounds."
                    )
                    success = True
                    break

            if not success:
                max_generations = self.cfg.task_generation.get(
                    "max_reflections", 0
                )
                print(
                    f"Task {task_number}: Task generation did not converge after {max_generations} rounds."
                )
                return None, []
            else:
                return response, (
                    message_history if message_history is not None else []
                )
        except Exception as e:
            self.logger.error(
                f"Error reflecting on task: {e}, input response: {response}"
            )
            return None, []

    def generate_initial_task_from_seed(
        self,
        seed_tasks_dir: str,
        task_number: int,  # Renamed from generation_index
        novel_init_adapt_rng_value: float,
    ) -> Optional[str]:
        """
        Generates a single task based on a seed task, without reflection.

        Args:
            seed_tasks_dir: Path to the seed task directory.
            task_number: The global task counter value for this task.
        """
        self.logger.debug(
            f"Generating task {task_number} from seed: {seed_tasks_dir}"
        )
        try:
            seed_py_file = os.path.join(seed_tasks_dir, "task.py")
            seed_json_file = os.path.join(seed_tasks_dir, "task.json")

            if not os.path.exists(seed_py_file) or not os.path.exists(
                seed_json_file
            ):
                self.logger.error(
                    f"Seed task files not found in {seed_tasks_dir}"
                )
                return None

            with open(seed_py_file, "r") as f:
                seed_python = f.read()
            with open(seed_json_file, "r") as f:
                seed_json = json.load(f)

            seed_json["task_family"] = seed_python
            seed_json_str = json.dumps(seed_json, indent=4)

            # Decide which prompt to use based on configuration
            novel_prompt_probability = self.cfg.task_generation.get(
                "novel_prompt_probability", 0.2
            )

            # Random selection based on probability
            if novel_init_adapt_rng_value < novel_prompt_probability:
                # Use the completely novel generation prompt (20% by default)
                self.logger.info(
                    f"Using completely novel generation prompt for task {task_number}"
                )
                prompt = initial_task_gen_prompt_completely_novel.format(
                    prev_json=seed_json_str
                )
            else:
                # Use the adapt similar generation prompt (80% by default)
                self.logger.info(
                    f"Using adapt similar generation prompt for task {task_number}"
                )
                prompt = initial_task_gen_prompt_adapt_similar.format(
                    prev_json=seed_json_str
                )

            # Use the dynamically loaded prompt string
            system_msg = self.task_creation_system_prompt_str.format(
                num_rounds=1
            )
            response_content, message_history = self._get_scientist_response(
                prompt=prompt, system_msg=system_msg, msg_history=None
            )

            if not response_content:
                self.logger.error("Scientist LLM returned no content.")
                return None

            response_json = extract_json_between_markers(response_content)
            if not response_json:
                self.logger.error(
                    f"Failed to extract JSON from scientist response: {response_content[:500]}..."
                )
                return None

            if not self._validate_generated_task(response_json):
                self.logger.error(
                    f"Generated task failed validation: {response_json}"
                )
                return None

            # --- Reflection Step ---
            if self.cfg.task_generation.get("max_reflections", 0) > 0:
                response_json, _ = self._reflect_on_task(
                    response_json, task_number, message_history
                )
            # --- End Reflection Step ---

            if response_json is None:
                self.logger.error(
                    f"Failed to generate task {task_number} from seed {seed_tasks_dir}"
                )
                return None

            task_name = response_json["name_of_task"]
            sanitized_name = sanitize_filename(task_name)
            # Use the global task_number passed as argument
            new_task_dir_name = f"task_{task_number}_{sanitized_name}"
            new_task_dir = os.path.join(
                self.generated_tasks_dir, new_task_dir_name
            )

            # Collision handling (should be less likely with global counter, but keep for safety)
            count = 0
            base_dir = new_task_dir
            while os.path.exists(new_task_dir):
                count += 1
                new_task_dir = f"{base_dir}_{count}"
                if count > 10:
                    self.logger.error(
                        f"Too many name collisions for {base_dir}"
                    )
                    return None

            pathlib.Path(new_task_dir).mkdir(parents=True, exist_ok=False)

            metadata = {
                "generated_from_seed": os.path.basename(seed_tasks_dir),
                "task_number": task_number,  # Use the global counter value
                "generation_type": "initial",
            }
            save_task_to_disk(new_task_dir, response_json, metadata)
            self.logger.info(
                f"Generated and saved task files to: {new_task_dir}"
            )

            # --- Add Sandbox Validation Step ---
            original_task_dir = new_task_dir
            validated_task_dir, _, _, example_instruction = (
                self._sandbox_validate_task(new_task_dir, generate_answer=False)
            )
            if validated_task_dir is None:
                self.logger.error(
                    f"Sandbox validation failed for {original_task_dir}. Removing directory."
                )
                shutil.rmtree(original_task_dir, ignore_errors=True)
                return None
            # --- End Sandbox Validation Step ---
            return validated_task_dir

        except Exception as e:
            self.logger.exception(
                f"Error generating task from seed {seed_tasks_dir}: {e}"
            )
            return None

    def _sandbox_validate_task(
        self,
        task_dir: str,
        generate_answer: bool = False,
        timeout_seconds: int = 300,
    ) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
        """
        Validate the task by running it in the sandbox.

        Args:
            task_dir: Path to the task directory
            generate_answer: Whether to generate an answer using the scientist model
            timeout_seconds: Timeout in seconds (default: 5 minutes)

        Returns:
            Tuple[str, float, str, str]: The task directory, the score, the answer, and the example instruction.
        """

        def _validate_task_impl():
            self.logger.info(
                f"Validating generated task {task_dir} in sandbox..."
            )
            task_py_path = os.path.join(task_dir, "task.py")

            # Load task family to get example data
            task_family_cls = load_task_family(task_dir)
            task_instance = task_family_cls()
            tasks_data = task_instance.get_tasks()

            if not tasks_data:
                raise ValueError("TaskFamily.get_tasks() returned no data.")

            # Prepare dummy input for the score function
            first_task_key = next(iter(tasks_data))
            task_data_for_eval = tasks_data[first_task_key]
            all_tasks_data = task_instance.get_tasks()
            first_example_id = next(iter(all_tasks_data))
            first_example_data = all_tasks_data[first_example_id]
            prompt = task_instance.get_instructions(first_example_data)
            if generate_answer:
                # prompt = task_instance.get_evaluation_prompt()
                answer, _ = self._get_scientist_response(
                    prompt=prompt,
                    system_msg="You are a helpful assistant that solves tasks.",
                )
            else:
                answer = "dummy validation submission"
            input_data = {
                "task_data": task_data_for_eval,
                "answer": answer,
            }

            # Run the 'score' method in the sandbox
            # This primarily checks if the code runs without crashing
            score = run_task_in_sandbox(
                task_script_path=task_py_path,
                function_name="score",
                input_data=input_data,
                cfg=self.cfg,
            )

            self.logger.info(f"Sandbox validation successful for {task_dir}")
            # self.logger.info(f"Example instruction: {prompt}")
            return (
                task_dir,
                score,
                answer,
                prompt,
            )  # Task is valid, return its path

        try:
            # Use ThreadPoolExecutor with timeout to handle long-running tasks
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                future = executor.submit(_validate_task_impl)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    self.logger.warning(
                        f"Task validation timed out after {timeout_seconds} seconds for {task_dir}"
                    )
                    return (
                        None,
                        None,
                        f"Task validation timed out after {timeout_seconds} seconds",
                        None,
                    )

        except Exception as e:
            return None, None, str(e), None  # Signal failure

    def _initialize_task_worker(self, args):
        """Worker function for parallel initialization."""
        seed_dir, task_number, novel_init_adapt_rng_value = (
            args  # task_number is the global counter value
        )
        # Note: generate_initial_task_from_seed now includes sandbox validation
        return self.generate_initial_task_from_seed(
            seed_dir, task_number, novel_init_adapt_rng_value
        )

    def initialize_pool(self):
        """
        Generates the initial task pool from seed tasks up to the configured size using multiprocessing.
        Optionally grows the seed pool with micro batches of generated tasks during initialization.
        """
        self.logger.info(
            f"Initializing task pool from seeds in {self.seed_tasks_dir} to size {self.initial_pool_size} using multiprocessing."
        )
        seed_dirs = [
            os.path.join(self.seed_tasks_dir, d)
            for d in os.listdir(self.seed_tasks_dir)
            if os.path.isdir(os.path.join(self.seed_tasks_dir, d))
        ]

        if not seed_dirs:
            raise ValueError(
                f"No seed task directories found in {self.seed_tasks_dir}"
            )

        # Track seed task IDs to exclude from final archive
        self.seed_task_ids = set()

        # Check if we should grow seed pool with micro batches
        grow_with_micro_batches = self.cfg.task_generation.get(
            "grow_seed_pool_with_micro_batches", False
        )
        micro_batch_size = self.cfg.task_generation.get("micro_batch_size", 10)

        if grow_with_micro_batches:
            self.logger.info(
                f"Growing seed pool with micro batches enabled. Micro batch size: {micro_batch_size}"
            )
            # First add original seed tasks to track their IDs
            self._add_original_seed_tasks_to_tracking(seed_dirs)

        num_workers = self.cfg.acdc.get(
            "num_initialization_workers", multiprocessing.cpu_count()
        )
        self.logger.info(f"Using {num_workers} workers for initialization.")

        generated_task_paths = []
        attempts = 0
        max_total_attempts = self.cfg.acdc.get(
            "max_total_initialization_attempts",
            self.initial_pool_size * 3,
        )  # Allow more attempts overall

        # --- Novelty filtering logic for initial pool ---
        queue_task_dirs = []  # All valid generated task dirs (for queue)
        main_pool_ids = set()  # For duplicate check

        with multiprocessing.Pool(processes=num_workers) as pool:
            current_seed_dirs = (
                seed_dirs.copy() if grow_with_micro_batches else seed_dirs
            )

            while (
                len(generated_task_paths) < self.initial_pool_size
                and attempts < max_total_attempts
            ):
                needed = self.initial_pool_size - len(generated_task_paths)

                # Use micro batch size when enabled, otherwise use dynamic batch size
                if grow_with_micro_batches:
                    batch_size = min(micro_batch_size, needed)
                    self.logger.info(
                        f"Generating micro batch of {batch_size} tasks (Need {needed} more, Total attempts: {attempts}/{max_total_attempts})"
                    )
                else:
                    # Generate slightly more than needed in each batch to account for failures
                    batch_size = min(needed * 2, num_workers * 2)
                    self.logger.info(
                        f"Attempting to generate batch of {batch_size} tasks (Need {needed} more, Total attempts: {attempts}/{max_total_attempts})"
                    )

                batch_args = []
                for i in range(batch_size):
                    seed_to_use = self.random_instance.choice(current_seed_dirs)
                    novel_init_adapt_rng_value = self.random_instance.random()
                    self.task_counter += 1
                    task_number = self.task_counter
                    batch_args.append(
                        (seed_to_use, task_number, novel_init_adapt_rng_value)
                    )
                    attempts += 1

                results = pool.map(self._initialize_task_worker, batch_args)

                # Process results and add to generated task paths
                successful_in_batch = 0
                batch_task_paths = []
                for task_path in results:
                    if task_path:
                        if task_path not in generated_task_paths:
                            generated_task_paths.append(task_path)
                            batch_task_paths.append(task_path)
                            successful_in_batch += 1
                        else:
                            self.logger.warning(
                                f"Duplicate task generated and skipped: {task_path}"
                            )

                self.logger.info(
                    f"Batch finished. Successfully generated: {successful_in_batch}/{len(results)}. Total valid tasks: {len(generated_task_paths)}/{self.initial_pool_size}. Current task counter: {self.task_counter}"
                )

                # Process the entire micro batch for seed pool growth
                if grow_with_micro_batches and batch_task_paths:
                    self._process_micro_batch_for_seed_growth(
                        batch_task_paths, current_seed_dirs
                    )

        # --- Truncate if over initial_pool_size ---
        if len(generated_task_paths) > self.initial_pool_size:
            self.logger.warning(
                f"Generated {len(generated_task_paths)} tasks, exceeding initial_pool_size ({self.initial_pool_size}). Truncating to keep the earliest generated tasks."
            )
            # Sort by task number (extracted from path) to ensure we keep the oldest
            task_pattern = re.compile(r"^task_(\d+)_.*")

            def get_task_num_init(path):
                match = task_pattern.match(os.path.basename(path))
                return int(match.group(1)) if match else float("inf")

            generated_task_paths.sort(key=get_task_num_init)
            generated_task_paths = generated_task_paths[
                : self.initial_pool_size
            ]
        # --- End Truncation ---

        # --- (Optional) Adapt initial task pool using seed models ---
        if self.cfg.acdc.get("adapt_initial_pool", False):
            raise NotImplementedError(
                "Adapt initial task pool not implemented yet."
            )

        # Validation step (instantiating ACDCTask) - kept sequential for simplicity
        validated_tasks = []
        for task_path in generated_task_paths:  # Use potentially truncated list
            try:
                # Attempt to load the task to validate it (final check)
                _ = ACDCTask(task_dir=task_path, cfg=self.cfg)
                validated_tasks.append(task_path)
            except Exception as e:
                self.logger.error(
                    f"Generated task at {task_path} failed final validation and will be removed: {e}"
                )

        if self.cfg.task_generation.use_init_queue_novelty_filtering:
            for task_path in validated_tasks:
                if not task_path:
                    continue
                if len(self.tasks) >= self.cfg.acdc.get("max_pool_size", 1000):
                    break
                task_code, relevant_metadata, task_id, task_json = (
                    self._load_task_code_and_metadata(task_path)
                )
                if task_id in main_pool_ids:  # TODO: maybe remove
                    continue  # Already added
                # Always add the first valid task
                if len(self.tasks) == 0:
                    self.add_task_and_vector_db(
                        task_path=task_path,
                        code=task_code,
                        metadata=relevant_metadata,
                        custom_id=task_id,
                        vector_dbs=[
                            self.vector_db_active,
                            self.vector_db_historical,
                        ],
                    )
                    main_pool_ids.add(task_id)
                    self.logger.info(
                        f"First task {task_id} added to main pool and vector DB."
                    )
                    continue
                # For subsequent tasks, check novelty
                similar_tasks = []
                if self.vector_db_historical is not None:
                    similar_tasks = self.vector_db_historical.find_similar(
                        query=task_code,
                        metadata=relevant_metadata,
                        top_n=self.cfg.task_generation.get(
                            "max_similar_tasks", 1
                        ),
                        similarity_threshold=0.0,
                    )
                is_novel = True
                if self.cfg.task_generation.get(
                    "interestingly_new_llm_check", False
                ):
                    is_novel = self._llm_based_interestingly_new_check(
                        task_json, similar_tasks
                    )
                elif not similar_tasks or len(similar_tasks) == 0:
                    is_novel = True
                else:
                    is_novel = False
                if is_novel:
                    self.add_task_and_vector_db(
                        task_path=task_path,
                        code=task_code,
                        metadata=relevant_metadata,
                        custom_id=task_id,
                        vector_dbs=[
                            self.vector_db_active,
                            self.vector_db_historical,
                        ],
                    )
                    main_pool_ids.add(task_id)
                    self.logger.info(
                        f"Novel task {task_id} added to main pool and vector DB."
                    )
                else:
                    # TODO: track the number of models discarded due to novelty check
                    self.logger.info(
                        f"Task {task_id} NOT added to main pool: failed novelty check."
                    )
                    # Build up historical archive to prevent similar tasks from appearing again
                    if self.vector_db_historical is not None:
                        self.vector_db_historical.add_sample(
                            content=task_code,
                            metadata=relevant_metadata,
                            custom_id=task_id,
                        )
        else:
            # --- Add all validated tasks to pool and vector DBs here ---
            for task_path in validated_tasks:
                task_code, relevant_metadata, task_id, _ = (
                    self._load_task_code_and_metadata(task_path)
                )
                # Only add if not already present
                if task_path not in self.tasks:
                    self.add_task_and_vector_db(
                        task_path=task_path,
                        code=task_code,
                        metadata=relevant_metadata,
                        custom_id=task_id,
                        vector_dbs=[
                            self.vector_db_active,
                            self.vector_db_historical,
                        ],
                    )

        if len(self.tasks) < self.initial_pool_size:
            self.logger.warning(
                f"Failed to generate the target initial pool size. Generated and validated {len(self.tasks)}/{self.initial_pool_size} tasks after {attempts} total generation attempts."
            )
        else:
            self.logger.info(
                f"Successfully generated and validated initial task pool with {len(self.tasks)} tasks."
            )

    def _add_original_seed_tasks_to_tracking(self, seed_dirs):
        """
        Track original seed task IDs to exclude them from the final archive.
        """
        for seed_dir in seed_dirs:
            try:
                seed_json_file = os.path.join(seed_dir, "task.json")
                if os.path.exists(seed_json_file):
                    with open(seed_json_file, "r") as f:
                        seed_json = json.load(f)
                    seed_task_id = seed_json.get(
                        "task_id", os.path.basename(seed_dir)
                    )
                    self.seed_task_ids.add(seed_task_id)
                    self.logger.debug(
                        f"Tracking original seed task ID: {seed_task_id}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Failed to track seed task {seed_dir}: {e}"
                )

    def _process_micro_batch_for_seed_growth(
        self, candidate_task_paths, current_seed_dirs
    ):
        """
        Process a micro batch of generated tasks for potential addition to the seed pool.
        Tasks go through novelty check and reflection before being added.
        """
        if not candidate_task_paths:
            return

        self.logger.info(
            f"Processing micro batch of {len(candidate_task_paths)} tasks for seed pool growth"
        )

        # Validate and filter tasks through novelty check and reflection
        validated_candidates = []
        for task_path in candidate_task_paths:
            try:
                # Load task and validate
                _ = ACDCTask(task_dir=task_path, cfg=self.cfg)

                # Check novelty if vector DB is available
                disable_novelty_filter = self.cfg.task_generation.get("disable_novelty_filter", False)
                if self.vector_db_active is not None or not disable_novelty_filter:
                    task_code, relevant_metadata, task_id, task_json = (
                        self._load_task_code_and_metadata(task_path)
                    )

                    similar_tasks = self.vector_db_active.find_similar(
                        query=task_code,
                        metadata=relevant_metadata,
                        top_n=self.cfg.task_generation.get(
                            "max_similar_tasks", 3
                        ),
                        similarity_threshold=0.0,
                    )

                    is_novel = True
                    if self.cfg.task_generation.get(
                        "interestingly_new_llm_check", False
                    ):
                        is_novel = self._llm_based_interestingly_new_check(
                            task_json, similar_tasks
                        )
                    elif similar_tasks and len(similar_tasks) > 0:
                        is_novel = False

                    if is_novel:
                        validated_candidates.append(task_path)
                        self.logger.debug(
                            f"Task {task_id} passed novelty check for seed pool growth"
                        )
                    else:
                        self.logger.debug(
                            f"Task {task_id} failed novelty check for seed pool growth"
                        )
                else:
                    validated_candidates.append(task_path)

            except Exception as e:
                self.logger.warning(
                    f"Task {task_path} failed validation for seed pool growth: {e}"
                )

        # Add validated candidates to current seed dirs for future generation
        for task_path in validated_candidates:
            current_seed_dirs.append(task_path)
            self.logger.info(
                f"Added {os.path.basename(task_path)} to seed pool for subsequent generations"
            )

    def load_existing_tasks(self):
        """
        Loads paths of existing generated tasks from the generated_tasks_dir.
        If restarting from checkpoint, loads the active task pool from the latest generation.
        """
        self.logger.info(
            f"Scanning for existing tasks in {self.generated_tasks_dir}"
        )
        loaded_task_paths = []
        max_task_num = -1

        # Check if we're loading from a checkpoint with active_pool_gen_*.json files
        checkpoint_pool_loaded = False
        if self.cfg.get("restart_dir"):
            checkpoint_pool_loaded = self._load_from_checkpoint_pool()

        if not checkpoint_pool_loaded:
            # Fallback to scanning directory for all task directories
            if os.path.exists(self.generated_tasks_dir):
                task_pattern = re.compile(r"^task_(\d+)_.*")
                for item in os.listdir(self.generated_tasks_dir):
                    item_path = os.path.join(self.generated_tasks_dir, item)
                    if (
                        os.path.isdir(item_path)
                        and os.path.exists(os.path.join(item_path, "task.py"))
                        and os.path.exists(os.path.join(item_path, "task.json"))
                    ):
                        loaded_task_paths.append(item_path)
                        # Extract task number to update counter
                        match = task_pattern.match(item)
                        if match:
                            task_num = int(match.group(1))
                            max_task_num = max(max_task_num, task_num)

            # Sort loaded tasks by task number before assigning
            def get_task_num(task_path):
                task_pattern = re.compile(r"^task_(\d+)_.*")
                match = task_pattern.match(os.path.basename(task_path))
                return int(match.group(1)) if match else float("inf")

            loaded_task_paths.sort(key=get_task_num)
            self.tasks = loaded_task_paths  # Assign sorted paths
            # Set the counter to the highest found number + 1 to avoid collisions
            self.task_counter = max_task_num + 1
            self.logger.info(
                f"Found {len(self.tasks)} existing tasks. Task counter reset to {self.task_counter}."
            )

        # Vector DBs are loaded from checkpoint during initialization if restart_dir is set

    def _load_from_checkpoint_pool(self) -> bool:
        """
        Load tasks from the latest active_pool_gen_*.json file in checkpoint.

        Returns:
            bool: True if successfully loaded from checkpoint, False otherwise
        """
        try:
            # Find the latest active_pool_gen_*.json file
            pool_files = []
            max_task_num = -1
            if os.path.exists(self.generated_tasks_dir):
                for file in os.listdir(self.generated_tasks_dir):
                    # Extract task number to update counter
                    task_pattern = re.compile(r"^task_(\d+)_.*")
                    match = task_pattern.match(file)
                    if match:
                        task_num = int(match.group(1))
                        max_task_num = max(max_task_num, task_num)
                    if file.startswith("active_pool_gen_") and file.endswith(
                        ".json"
                    ):
                        match = re.match(r"active_pool_gen_(\d+)\.json", file)
                        if match:
                            gen_num = int(match.group(1))
                            pool_files.append((gen_num, file))

            if not pool_files:
                self.logger.info(
                    "No active_pool_gen_*.json files found in checkpoint"
                )
                return False

            # Sort by generation number and get the latest
            pool_files.sort(key=lambda x: x[0], reverse=True)
            latest_gen, latest_file = pool_files[0]

            # Load the latest active pool
            latest_pool_path = os.path.join(
                self.generated_tasks_dir, latest_file
            )
            with open(latest_pool_path, "r") as f:
                active_task_paths = json.load(f)

            # Validate that all task paths exist and have required files
            validated_paths = []

            for task_path in active_task_paths:
                if (
                    os.path.isdir(task_path)
                    and os.path.exists(os.path.join(task_path, "task.py"))
                    and os.path.exists(os.path.join(task_path, "task.json"))
                ):
                    validated_paths.append(task_path)
                else:
                    self.logger.warning(
                        f"Task path from checkpoint not found or invalid: {task_path}"
                    )

            self.tasks = validated_paths
            self.task_counter = max_task_num + 1

            # Load active limbo map if it exists for this generation
            limbo_map_file = f"active_limbo_map_gen_{latest_gen}.json"
            limbo_map_path = os.path.join(
                self.generated_tasks_dir, limbo_map_file
            )

            if os.path.exists(limbo_map_path):
                try:
                    with open(limbo_map_path, "r") as f:
                        self.active_limbo_map = json.load(f)
                    self.logger.info(
                        f"Loaded active limbo map with {len(self.active_limbo_map)} entries from {limbo_map_file}."
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load active limbo map from {limbo_map_path}: {e}"
                    )
                    self.active_limbo_map = (
                        {}
                    )  # Reset to empty dict if loading fails
            else:
                self.logger.info(
                    f"No active limbo map file found for generation {latest_gen}, starting with empty limbo map."
                )
                self.active_limbo_map = {}

            self.logger.info(
                f"Loaded {len(validated_paths)} tasks from checkpoint pool (generation {latest_gen}). "
                f"Task counter set to {self.task_counter}."
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading from checkpoint pool: {e}")
            return False

    def get_tasks(self) -> List[ACDCTask]:
        """
        Loads task objects from the generated task directories.
        """
        loaded_tasks = []
        for task_dir in self.tasks:
            try:
                task_obj = ACDCTask(task_dir=task_dir, cfg=self.cfg)
                loaded_tasks.append(task_obj)
            except Exception as e:
                self.logger.error(
                    f"Failed to load task from directory {task_dir}: {e}"
                )
        self.logger.info(f"Loaded {len(loaded_tasks)} task objects.")
        return loaded_tasks

    def get_ordered_task_ids(self) -> List[str]:
        """
        Returns a consistently ordered list of task IDs derived from the task directories.
        Currently uses the basename of the task directory as the ID and sorts alphabetically.
        """
        # Sort by task number extracted from the directory name for consistency
        task_pattern = re.compile(r"^task_(\d+)_.*")
        task_tuples = []
        for task_dir in self.tasks:
            match = task_pattern.match(os.path.basename(task_dir))
            if match:
                task_num = int(match.group(1))
                task_tuples.append((task_num, task_dir))
            else:
                self.logger.warning(
                    f"Could not extract task number from {task_dir}, using -1 for sorting."
                )
                task_tuples.append(
                    (-1, task_dir)
                )  # Assign -1 if pattern doesn't match

        task_tuples.sort(key=lambda x: x[0])  # Sort by task number (ascending)
        return [
            os.path.basename(t[1]) for t in task_tuples
        ]  # Return sorted basenames

    def _calculate_task_pass_rates(
        self, dns_archive: List[ACDCSolution]
    ) -> Tuple[
        Dict[str, float], Dict[str, int], Dict[str, int], Dict[str, str]
    ]:
        """Calculates pass rates for tasks currently in the pool based on archive performance."""
        task_pass_counts: Dict[str, int] = {}
        task_total_counts: Dict[str, int] = {}
        current_task_ids_paths = {
            os.path.basename(p): p for p in self.tasks
        }  # Map ID to path

        skill_threshold = self.cfg.dns.get("acdc_skill_threshold", 0.5)

        for solution in dns_archive:
            if solution and solution.acdc_skill_vector:
                for task_id, score in solution.acdc_skill_vector.items():
                    if (
                        task_id in current_task_ids_paths
                    ):  # Only consider tasks currently in the pool
                        task_total_counts[task_id] = (
                            task_total_counts.get(task_id, 0) + 1
                        )
                        if score >= skill_threshold:
                            task_pass_counts[task_id] = (
                                task_pass_counts.get(task_id, 0) + 1
                            )

        task_pass_rates: Dict[str, float] = {}
        for task_id, total_count in task_total_counts.items():
            if total_count > 0:
                pass_count = task_pass_counts.get(task_id, 0)
                task_pass_rates[task_id] = pass_count / total_count
            else:
                # Should not happen if task_id is in task_total_counts keys
                task_pass_rates[task_id] = (
                    -1.0
                )  # Indicate error or not evaluated

        # Add tasks from current_task_ids_paths that were not evaluated at all
        for task_id in current_task_ids_paths:
            if task_id not in task_pass_rates:
                task_pass_rates[task_id] = -1.0  # Indicate not evaluated

        self.logger.info(
            f"Calculated pass rates for {len(task_pass_rates)} tasks."
        )
        return (
            task_pass_rates,
            task_pass_counts,
            task_total_counts,
            current_task_ids_paths,
        )

    def _prioritize_tasks_for_adaptation(
        self,
        task_pass_rates: Dict[str, float],
        current_task_ids_paths: Dict[str, str],
    ) -> Tuple[
        List[Tuple[float, str]],
        List[Tuple[float, str]],
        List[Tuple[float, str]],
        List[Tuple[float, str]],
    ]:
        """Categorizes tasks based on pass rates and returns separate lists."""
        easy_tasks = []
        hard_tasks = []
        impossible_tasks = []
        medium_tasks = []  # Renamed from other_tasks for clarity
        unevaluated_tasks = []

        easy_threshold = self.cfg.acdc.get("difficulty_threshold_easy", 0.7)
        hard_threshold = self.cfg.acdc.get("difficulty_threshold_hard", 0.3)

        for task_id, task_path in current_task_ids_paths.items():
            pass_rate = task_pass_rates.get(task_id, -1.0)
            task_tuple = (pass_rate, task_path)

            if pass_rate == -1.0:  # Unevaluated tasks
                unevaluated_tasks.append(
                    task_tuple
                )  # Keep in separate list for now
            elif pass_rate == 0.0:
                impossible_tasks.append(task_tuple)
            elif pass_rate < hard_threshold:
                hard_tasks.append(task_tuple)
            elif pass_rate > easy_threshold:
                easy_tasks.append(task_tuple)
            else:  # Medium difficulty tasks
                medium_tasks.append(task_tuple)  # Add medium tasks here

        # Sort for prioritization / determinism
        impossible_tasks.sort(
            key=lambda x: x[1]
        )  # Sort by path for determinism
        hard_tasks.sort(
            key=lambda x: (x[0], x[1])
        )  # Sort by rate (asc), then path
        easy_tasks.sort(
            key=lambda x: (-x[0], x[1])
        )  # Sort by rate (desc), then path
        medium_tasks.sort(key=lambda x: (x[0], x[1]))  # Sort by rate, then path
        unevaluated_tasks.sort(key=lambda x: x[1])  # Sort by path

        # Combine medium and unevaluated for the 'other' category if needed later, or return separately
        # For now, return them separately as per plan (medium_tasks is the 4th list)
        self.logger.info(
            f"Categorized tasks: {len(impossible_tasks)} impossible, {len(hard_tasks)} hard, "
            f"{len(easy_tasks)} easy, {len(medium_tasks)} medium, {len(unevaluated_tasks)} unevaluated."
        )
        # Return the categorized lists as per the plan
        return (
            impossible_tasks,
            hard_tasks,
            easy_tasks,
            medium_tasks,
        )  # Ignoring unevaluated for selection for now

    def _determine_adaptation_type_and_prompt(
        self, original_pass_rate: float, original_task_path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determines the adaptation type (easy, hard, novel) and the corresponding prompt template."""
        easy_threshold = self.cfg.acdc.get("difficulty_threshold_easy", 0.7)
        hard_threshold = self.cfg.acdc.get("difficulty_threshold_hard", 0.3)
        completely_novel_prompt_probability = self.cfg.task_generation.get(
            "novel_prompt_probability", 0.2
        )
        make_novel_not_harder_prompt_probability = self.cfg.task_generation.get(
            "make_novel_not_harder_prompt_probability", 0.8
        )

        if original_pass_rate == 0.0:
            prompt_template = make_task_easier_prompt
            adapt_type = "easier"
        elif original_pass_rate < hard_threshold:
            prompt_template = make_task_easier_prompt
            adapt_type = "easier"
        elif original_pass_rate > easy_threshold:
            # TODO: Discuss this with the team
            if (
                self.random_instance.random()
                < make_novel_not_harder_prompt_probability
            ):
                prompt_template = make_task_novel_prompt
                adapt_type = "novel"
            else:
                prompt_template = make_task_harder_prompt
                adapt_type = "harder"
        else:  # Tasks in the middle range
            if (
                self.random_instance.random()
                < completely_novel_prompt_probability
            ):
                # Use the completely novel generation prompt (20% by default)
                prompt_template = make_task_novel_but_similar_prompt
            else:
                # Use the novel but similar generation prompt (80% by default)
                prompt_template = make_task_novel_prompt
            adapt_type = "novel"

        self.logger.info(
            f"Adaptation type for {os.path.basename(original_task_path)}: {adapt_type}"
        )
        return adapt_type, prompt_template

    def _llm_based_interestingly_new_check(
        self,
        new_task_json: Dict,
        closest_tasks: List[Dict[str, Any]],
    ) -> bool:
        new_task_description_str = self._format_task_description(new_task_json)
        closest_tasks_description_str = []
        for i, task in enumerate(closest_tasks):
            closest_tasks_description_str.append(
                f"Task {i+1}:\n{self._format_task_description(task['metadata'])}"
            )
        closest_tasks_description_str = "\n".join(closest_tasks_description_str)
        prompt = interestingly_new_prompt.format(
            new_task=new_task_description_str,
            closest_tasks=closest_tasks_description_str,
        )
        response_content, _ = self._get_scientist_response(
            prompt=prompt,
            system_msg=interestingly_new_system_msg,
        )

        # Extract the decision from the response "Decision: Yes" or "Decision: No"
        if response_content is None:
            self.logger.error("No response content received for decision")
            return False

        re_pattern = r"Decision: (Yes|No)"
        match = re.search(re_pattern, response_content, re.DOTALL)
        if match:
            decision = match.group(1)
        else:
            self.logger.error(
                f"No decision found in response: {response_content}"
            )
            return False

        return decision == "Yes"

    def _format_task_description(self, task_json: Dict) -> str:
        task_description = ""
        for key, value in task_json.items():
            if key == "task_family" or key == "done":
                continue
            key = key.capitalize().replace("_", " ")
            task_description += f"{key}: {value}\n"
        return task_description

    def _get_random_pool_tasks_for_prompt(self) -> List[Dict]:
        """
        Randomly select tasks from self.tasks and format them similar to vector DB results.

        Returns:
            List of task dictionaries with 'content', 'metadata', and 'sample_id' keys,
            similar to the format returned by vector DB find_similar.
        """
        max_similar_tasks = self.cfg.task_generation.get("max_similar_tasks", 3)

        if not self.tasks:
            self.logger.warning(
                "No tasks available in self.tasks for random selection"
            )
            return []

        # Determine how many tasks to select
        num_to_select = min(len(self.tasks), max_similar_tasks)

        # Apply randomization if enabled
        if self.cfg.task_generation.get("randomize_similar_tasks", False):
            selected_task_paths = self.random_instance.sample(
                self.tasks, num_to_select
            )
        else:
            # Take first N tasks if randomization is disabled
            selected_task_paths = self.tasks[:num_to_select]

        similar_tasks = []
        for task_path in selected_task_paths:
            try:
                task_code, relevant_metadata, task_id, task_json = (
                    self._load_task_code_and_metadata(task_path)
                )

                # Format similar to vector DB result
                similar_task = {
                    "content": task_code,
                    "metadata": task_json,
                    "sample_id": task_id,
                    "similarity": 1.0,  # Dummy similarity score since we're not using actual similarity
                }
                similar_tasks.append(similar_task)

            except Exception as e:
                self.logger.warning(
                    f"Failed to load task data from {task_path}: {e}"
                )
                continue

        # self.logger.info(f"Selected {len(similar_tasks)} random tasks from pool for prompt")
        return similar_tasks

    def _ensure_novel_task(
        self,
        original_json: Dict,
        prompt_template: str,
        system_msg: str,
        original_task_path: str,
    ) -> Tuple[Optional[str], Optional[List]]:
        """Ensure the task is novel by checking the vector db."""

        # Get original python code
        original_python = original_json["task_family"]
        original_json_str = json.dumps(original_json, indent=4)

        relevant_metadata = {
            k: v
            for k, v in original_json.items()
            if k != "task_family" and k != "done"
        }

        # Get similar tasks - either from vector DB or randomly from task pool
        if self.cfg.task_generation.get("use_random_pool_tasks", False):
            # Randomly select tasks from self.tasks instead of using vector DB
            similar_tasks = self._get_random_pool_tasks_for_prompt()
        else:
            # Get similar tasks from active vector db
            if self.vector_db_active is not None:
                similar_tasks = self.vector_db_active.find_similar(
                    query=original_python,
                    metadata=relevant_metadata,
                    top_n=self.cfg.task_generation.get("max_similar_tasks", 1),
                    similarity_threshold=0.0,
                )
            else:
                similar_tasks = []

        task_representation = self.cfg.task_generation.get(
            "task_representation", "metadata"
        )
        similar_tasks_str = ""
        for task in similar_tasks:
            if task_representation == "metadata":
                task_metadata = self._format_task_description(task["metadata"])
                similar_tasks_str += "[[START OF ADDITIONAL EXISTING TASK]]\n"
                similar_tasks_str += f"Task description: \n{task_metadata}\n"
                similar_tasks_str += "[[END OF ADDITIONAL EXISTING TASK]]\n\n"
            # Use task python code as representation
            elif task_representation == "content":
                task_content = task["content"]
                similar_tasks_str += "[[START OF ADDITIONAL EXISTING TASK]]\n"
                similar_tasks_str += f"Task code: \n{task_content}\n"
                similar_tasks_str += "[[END OF ADDITIONAL EXISTING TASK]]\n\n"
            # Use both metadata and content
            elif task_representation == "all":
                task_metadata = self._format_task_description(task["metadata"])
                similar_tasks_str += "[[START OF ADDITIONAL EXISTING TASK]]\n"
                similar_tasks_str += f"Task description: \n{task_metadata}\n"
                task_content = task["content"]
                similar_tasks_str += f"Task code: \n{task_content}\n"
                similar_tasks_str += "[[END OF ADDITIONAL EXISTING TASK]]\n\n"

        prompt = prompt_template.format(
            original_task_json=original_json_str,
            other_task_jsons=similar_tasks_str,
        )

        is_novel = False
        # max_retries = self.cfg.task_generation.get("max_novelty_reprompting", 1)
        # retries = 0
        # message_history = None
        # while not is_novel and retries < max_retries:
        # Get response from scientist - the proposed new task
        response_content, message_history = self._get_scientist_response(
            prompt=prompt,
            system_msg=system_msg,
            msg_history=None,
        )

        if not response_content:
            self.logger.error(
                f"Scientist LLM returned no content for adapting {original_task_path}."
            )
            return None, None

        response_json = extract_json_between_markers(response_content)
        if not response_json:
            self.logger.error(
                f"Failed to extract JSON for adapting {original_task_path}."
            )
            return None, None

        # Extract metadata and content from response_json
        content = response_json["task_family"]

        relevant_metadata = {
            k: v
            for k, v in response_json.items()
            if k != "task_family" and k != "done"
        }

        # Try to find similar tasks in the historical vector db
        if self.vector_db_historical is not None:
            similar_tasks_historical = self.vector_db_historical.find_similar(
                query=content,
                metadata=relevant_metadata,
                top_n=self.cfg.task_generation.get("max_similar_tasks", 1),
                similarity_threshold=self.cfg.task_generation.get(
                    "similarity_threshold", 0.0
                ),
            )
        else:
            similar_tasks_historical = None

        # Check if the task is novel using the LLM-based check
        if (
            self.cfg.task_generation.get("interestingly_new_llm_check", False)
            and similar_tasks_historical is not None
        ):
            is_novel = self._llm_based_interestingly_new_check(
                response_json, similar_tasks_historical
            )
        # If the LLM-based check is not used, use the embedding-based check
        elif not similar_tasks_historical or len(similar_tasks_historical) == 0:
            is_novel = True

        # If the task is still not novel and we use hard filtering, discard it
        # TODO: track the number of models discarded due to novelty check
        if not is_novel and self.cfg.task_generation.get(
            "hard_novelty_filtering", False
        ):
            self.logger.info(
                f"Using hard filtering to ensure novel task. The task {original_task_path} was not novel. Discarding..."
            )
            return None, None

        return response_content, message_history

    def _generate_and_validate_adapted_task(  # Signature updated
        self,
        original_task_path: str,
        original_pass_rate: float,
        task_number: int,  # Added task_number calculated by caller
        adapt_type: str,  # Pass adapt_type determined by caller
        prompt_template: str,  # Pass prompt_template determined by caller
    ) -> Optional[str]:
        """Generates, saves, and validates a single adapted task."""
        # adapt_type and prompt_template are now determined and passed by the caller in adapt_task_pool
        # The check for None is handled there before calling this function.

        # Use the dynamically loaded prompt string
        system_msg = self.task_creation_system_prompt_str.format(
            num_rounds=self.cfg.task_generation.get("max_reflections", 1)
        )  # No reflection

        try:
            # Load original task details
            original_py_file = os.path.join(original_task_path, "task.py")
            original_json_file = os.path.join(original_task_path, "task.json")
            if not os.path.exists(original_py_file) or not os.path.exists(
                original_json_file
            ):
                self.logger.error(
                    f"Original task files not found for {original_task_path}. Skipping."
                )
                return None

            with open(original_py_file, "r") as f:
                original_python = f.read()
            with open(original_json_file, "r") as f:
                original_json = json.load(f)

            original_json["task_family"] = original_python
            original_json_str = json.dumps(original_json, indent=4)

            # --- Novelty Reprompting Step ---
            do_novelty_search = (
                self.cfg.task_generation.get(
                    "novelty_search_all_adapt_types", True
                )
                or adapt_type == "novel"
            )
            if (
                self.vector_db_active is not None
                and self.vector_db_historical is not None
                and do_novelty_search
            ):
                response_content, _ = self._ensure_novel_task(
                    original_json,
                    prompt_template,
                    system_msg,
                    original_task_path,
                )
                message_history = None
                if not response_content:
                    self.logger.error(
                        f"Failed to ensure novel task for {original_task_path}."
                    )
                    return None
            # --- End Novelty Reprompting Step ---
            else:
                # Format prompt
                prompt = prompt_template.format(
                    original_task_json=original_json_str, other_task_jsons=""
                )

                # Get response from scientist
                response_content, message_history = (
                    self._get_scientist_response(prompt, system_msg)
                )
                if not response_content:
                    self.logger.error(
                        f"Scientist LLM returned no content for adapting {original_task_path}."
                    )
                    return None

            response_json = extract_json_between_markers(response_content)
            if not response_json:
                self.logger.error(
                    f"Failed to extract JSON for adapting {original_task_path}."
                )
                return None

            # --- Reflection Step ---
            if self.cfg.task_generation.get("max_reflections", 0) > 0:
                reflected_response, _ = self._reflect_on_task(
                    response_json, task_number, message_history
                )
                if reflected_response is None:
                    self.logger.error(
                        f"Reflection failed for adapted task from {original_task_path}."
                    )
                    return None
                response_json = reflected_response
            # --- End Reflection Step ---

            if not self._validate_generated_task(response_json):
                self.logger.error(
                    f"Generated adapted task failed validation for {original_task_path}."
                )
                return None

            # Create new directory and save
            task_name = response_json["name_of_task"]
            sanitized_name = sanitize_filename(task_name)
            # Use the global task_number passed as argument
            new_task_dir_name = f"task_{task_number}_{sanitized_name}"
            new_task_dir = os.path.join(
                self.generated_tasks_dir, new_task_dir_name
            )

            # Collision handling (should be less likely with global counter, but keep for safety)
            count = 0
            base_dir = new_task_dir
            while os.path.exists(new_task_dir):
                count += 1
                new_task_dir = f"{base_dir}_{count}"
                if count > 10:
                    self.logger.error(
                        f"Too many name collisions for adapted task {base_dir}"
                    )
                    return None  # Indicate failure due to collisions

            pathlib.Path(new_task_dir).mkdir(parents=True, exist_ok=False)

            metadata = {
                "generated_from_task": os.path.basename(original_task_path),
                "task_number": task_number,  # Use the global counter value
                "generation_type": f"adaptation_{adapt_type}",
                "original_pass_rate": original_pass_rate,
            }
            save_task_to_disk(new_task_dir, response_json, metadata)
            self.logger.info(
                f"Generated and saved adapted task to: {new_task_dir}"
            )

            # Sandbox Validate
            # --- Sandbox Validation Step ---
            original_task_dir = (
                new_task_dir  # Store original path before validation
            )
            try:
                validated_task_dir, _, _, example_instruction = (
                    self._sandbox_validate_task(
                        new_task_dir, generate_answer=False
                    )
                )
                if validated_task_dir is None:
                    self.logger.error(
                        f"Sandbox validation failed for {original_task_dir}. Removing directory."
                    )
                    shutil.rmtree(original_task_dir, ignore_errors=True)
                    return None
                new_task_dir = validated_task_dir  # Update with validated path
                # else:
                #     # Use general add method for adapted tasks
                #     relevant_metadata = {
                #         k: v
                #         for k, v in response_json.items()
                #         if k != "task_family" and k != "done"
                #     }
                #     self.add_task_and_vector_db(
                #         task_path=new_task_dir,
                #         code=response_json["task_family"],
                #         metadata=relevant_metadata,
                #         custom_id=new_task_dir_name,
                #         vector_dbs=[self.vector_db_active, self.vector_db_historical]
                #     )
                #     return new_task_dir  # Return path on success
                return new_task_dir
            except Exception as e:
                self.logger.error(
                    f"Sandbox validation failed for adapted task {original_task_dir}. Removing directory. Error: {e}"
                )
                if original_task_dir and os.path.exists(original_task_dir):
                    shutil.rmtree(original_task_dir, ignore_errors=True)
                return None  # Return None on validation failure

        except Exception as e:
            self.logger.exception(
                f"Error during adaptation process for {original_task_path}: {e}"
            )
            return None

    def _update_pool_after_adaptation(
        self,
        newly_generated_tasks: List[str],
        successfully_replaced_originals: Dict[str, float],
    ) -> Tuple[int, List[str]]:
        """Updates the task pool list by adding new tasks and removing replaced impossible ones."""
        removed_impossible_count = 0
        other_originals_replaced = []
        current_tasks_set = set(self.tasks)

        # Separate replaced originals based on pass rate and remove impossible ones
        for (
            original_path,
            original_rate,
        ) in successfully_replaced_originals.items():
            if original_rate == 0.0:  # Impossible task
                # Use general remove method for impossible tasks
                self.remove_task_and_vector_db(
                    original_path, vector_dbs=[self.vector_db_active]
                )
                self.logger.info(
                    f"Removed replaced impossible task from active pool: {os.path.basename(original_path)}"
                )
                removed_impossible_count += 1
            else:  # Other tasks (hard, easy, etc.) that were replaced
                other_originals_replaced.append(original_path)

        self.logger.info(
            f"Removed {removed_impossible_count} impossible tasks that were replaced."
        )

        # Add newly generated tasks
        added_count = 0
        for new_path in newly_generated_tasks:
            if new_path not in current_tasks_set:  # Avoid duplicates
                # No need to add to vector DB here, already handled at creation
                if new_path not in self.tasks:
                    task_code, relevant_metadata, task_id, _ = (
                        self._load_task_code_and_metadata(new_path)
                    )
                    self.add_task_and_vector_db(
                        task_path=new_path,
                        code=task_code,
                        metadata=relevant_metadata,
                        custom_id=task_id,
                        vector_dbs=[
                            self.vector_db_active,
                            self.vector_db_historical,
                        ],
                    )
                current_tasks_set.add(new_path)
                added_count += 1
        self.logger.info(f"Added {added_count} new tasks to active pool.")

        # Re-sort self.tasks by task number after additions/removals
        task_pattern = re.compile(
            r"^task_(\d+)_.*"
        )  # Define pattern if not accessible

        def get_task_num_update(task_path):
            match = task_pattern.match(os.path.basename(task_path))
            return int(match.group(1)) if match else float("inf")

        self.tasks.sort(key=get_task_num_update)
        self.logger.info(
            f"Task pool re-sorted by task number. Size before pruning: {len(self.tasks)}"
        )

        return removed_impossible_count, other_originals_replaced

    def _prune_task_pool_if_needed(
        self,
        other_originals_replaced: List[str],
        # task_pass_counts and task_total_counts are no longer needed here
        # as pruning is based on age (task number)
    ) -> int:
        """Prunes the task pool if it exceeds max size, prioritizing replaced originals, then oldest."""
        pruned_count = 0
        max_pool_size = self.cfg.acdc.get("max_pool_size", 1000)
        task_pattern = re.compile(
            r"^task_(\d+)_.*"
        )  # Regex to extract task number
        current_tasks_set = set(
            self.tasks
        )  # Keep track of current set for efficient checks

        # Pruning loop
        while len(self.tasks) > max_pool_size:
            if not self.tasks:
                self.logger.warning("Cannot prune further: Task pool is empty.")
                break

            # Extract task numbers and paths
            task_tuples = []
            for task_path in self.tasks:
                match = task_pattern.match(os.path.basename(task_path))
                if match:
                    task_num = int(match.group(1))
                    task_tuples.append((task_num, task_path))
                else:
                    # Handle tasks that don't match the pattern (e.g., manually added)
                    # Assign a high number so they are pruned last, or handle differently
                    self.logger.warning(
                        f"Could not extract task number from {task_path} during pruning. Assigning high number."
                    )
                    task_tuples.append((float("inf"), task_path))

            if not task_tuples:
                self.logger.error(
                    "Cannot prune further: Failed to extract task numbers."
                )
                break

            # Sort tasks by number (ascending) - oldest first
            task_tuples.sort(key=lambda x: x[0])

            # Get the oldest task to prune
            oldest_task_num, oldest_task_path = task_tuples[0]

            if oldest_task_path in current_tasks_set:
                # Use general remove method for pruning oldest task
                self.remove_task_and_vector_db(
                    oldest_task_path, vector_dbs=[self.vector_db_active]
                )
                self.logger.info(
                    f"Pruned oldest task from pool (Num: {oldest_task_num}): {os.path.basename(oldest_task_path)}"
                )
                pruned_count += 1
            else:
                self.logger.error(
                    f"Failed to find or remove oldest task '{os.path.basename(oldest_task_path)}' during pruning."
                )
                # Remove from task_tuples to avoid trying again
                task_tuples.pop(0)
                if not task_tuples:
                    break  # Exit if no more tasks to try

        if pruned_count > 0:
            self.logger.info(
                f"Pruned {pruned_count} tasks in total. Pool size now: {len(self.tasks)}"
            )

        return pruned_count

    def _adapt_task_worker(self, args):
        """Worker function for parallel adaptation."""
        # Updated worker arguments to include task_number, adapt_type, prompt_template
        (
            original_task_path,
            original_pass_rate,
            task_number,
            adapt_type,
            prompt_template,
        ) = args
        # Note: _generate_and_validate_adapted_task includes sandbox validation
        # Pass the new arguments to the generation function
        new_task_path = self._generate_and_validate_adapted_task(
            original_task_path=original_task_path,
            original_pass_rate=original_pass_rate,
            task_number=task_number,
            adapt_type=adapt_type,
            prompt_template=prompt_template,
        )
        # Return both the new path (or None) and the original path/rate for tracking
        # Also return the task_number for conditional parent replacement
        return (
            new_task_path,
            original_task_path,
            original_pass_rate,
            task_number,
        )

    def _get_tasks_for_adaptation_biased(
        self,
        hard_tasks: List[Tuple[float, str]],
        easy_tasks: List[Tuple[float, str]],
        medium_tasks: List[Tuple[float, str]],
        num_to_adapt: int,
        impossible_tasks: Optional[List[Tuple[float, str]]] = None,
    ) -> Tuple[List[Tuple[float, str]], int]:
        """
        Selects tasks for adaptation, prioritizing impossible tasks and balancing across difficulty levels.
        1. Select all impossible tasks for adaptation if not None.
        2. Select a random sample of easy/hard tasks for adaptation
        3. Select a random sample of medium tasks for adaptation

        Args:
            hard_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            easy_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            medium_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            num_to_adapt: int - Number of tasks to adapt
            impossible_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)

        Returns:
            selected_task_tuples: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            remaining_slots: int - Number of slots remaining after selecting tasks
        """
        selected_task_tuples = []
        remaining_slots = num_to_adapt

        discard_impossible_tasks = self.cfg.task_generation.get(
            "discard_impossible_tasks", True
        )

        def select_impossible_tasks(
            impossible_tasks: List[Tuple[float, str]], remaining_slots: int
        ) -> Tuple[List[Tuple[float, str]], int]:
            num_impossible = len(impossible_tasks)
            selected_impossible_task_tuples = []
            if num_impossible > 0 and remaining_slots > 0:
                if num_impossible >= remaining_slots:
                    # Select uniformly random sample if more impossible tasks than slots
                    selected_impossible = self.random_instance.sample(
                        impossible_tasks, remaining_slots
                    )
                    selected_impossible_task_tuples.extend(selected_impossible)
                    self.logger.info(
                        f"Selected {remaining_slots} impossible tasks (random sample)."
                    )
                    remaining_slots = 0
                else:
                    # Select all impossible tasks if fewer than remaining slots
                    selected_impossible_task_tuples.extend(impossible_tasks)
                    remaining_slots -= num_impossible
                    self.logger.info(
                        f"Selected all {num_impossible} impossible tasks."
                    )

            return selected_impossible_task_tuples, remaining_slots

        # --- Impossible Task Selection ---
        if not discard_impossible_tasks and impossible_tasks is not None:
            selected_impossible_task_tuples, remaining_slots = (
                select_impossible_tasks(
                    impossible_tasks=impossible_tasks,
                    remaining_slots=remaining_slots,
                )
            )
            selected_task_tuples.extend(selected_impossible_task_tuples)

        # --- Easy/Hard Task Selection (Weighted) ---
        if remaining_slots > 0:
            extreme_tasks = hard_tasks + easy_tasks
            num_extreme = len(extreme_tasks)
            if num_extreme > 0:
                num_to_select_extreme = min(remaining_slots, num_extreme)
                # Calculate weights based on distance from 0.5 pass rate
                # Add epsilon to avoid zero weights if pass rate is exactly 0.5 (unlikely for hard/easy)
                weights = [
                    abs(rate - 0.5) + 1e-6 for rate, path in extreme_tasks
                ]
                try:
                    # Ensure population and weights are not empty before calling choices
                    if extreme_tasks and weights and num_to_select_extreme > 0:
                        selected_extreme = self.random_instance.choices(
                            extreme_tasks,
                            weights=weights,
                            k=num_to_select_extreme,
                        )
                        selected_task_tuples.extend(selected_extreme)
                        remaining_slots -= num_to_select_extreme
                        self.logger.info(
                            f"Selected {num_to_select_extreme} easy/hard tasks (weighted)."
                        )
                    elif num_to_select_extreme == 0:
                        self.logger.info(
                            "No easy/hard tasks needed based on remaining slots."
                        )
                    else:
                        self.logger.warning(
                            "Cannot perform weighted choice with empty population or weights."
                        )

                except ValueError as e:
                    self.logger.error(
                        f"Error during weighted choice for extreme tasks: {e}. Weights: {weights}, k={num_to_select_extreme}, Population size: {len(extreme_tasks)}"
                    )
                    # Fallback: Select uniformly if weighted choice fails? Or just log? Logging for now.
            else:
                self.logger.info(
                    "No easy or hard tasks available for weighted selection."
                )

        # --- Medium Task Selection (Optional Fallback - Uniform Random) ---
        # If slots still remain after impossible and extreme, fill with medium.
        if remaining_slots > 0 and medium_tasks:
            num_medium = len(medium_tasks)
            num_to_select_medium = min(remaining_slots, num_medium)
            selected_medium = self.random_instance.sample(
                medium_tasks, num_to_select_medium
            )
            selected_task_tuples.extend(selected_medium)
            remaining_slots -= num_to_select_medium
            self.logger.info(
                f"Selected {num_to_select_medium} medium tasks (uniform random fallback)."
            )

        # --- Impossible Task Selection (Worst case fallback when there are no other tasks) ---
        if (
            len(selected_task_tuples) == 0
            and discard_impossible_tasks
            and impossible_tasks is not None
        ):
            selected_impossible_task_tuples, remaining_slots = (
                select_impossible_tasks(
                    impossible_tasks=impossible_tasks,
                    remaining_slots=remaining_slots,
                )
            )
            selected_task_tuples.extend(selected_impossible_task_tuples)

        return selected_task_tuples, remaining_slots

    def _get_tasks_for_adaptation_uniform(
        self,
        impossible_tasks: List[Tuple[float, str]],
        hard_tasks: List[Tuple[float, str]],
        easy_tasks: List[Tuple[float, str]],
        medium_tasks: List[Tuple[float, str]],
        num_to_adapt: int,
    ) -> Tuple[List[Tuple[float, str]], int]:
        """
        Selects tasks for adaptation uniformly at random.

        Args:
            hard_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            easy_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            medium_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            num_to_adapt: int - Number of tasks to adapt
            impossible_tasks: List[Tuple[float, str]] - List of task tuples (pass rate, task path)

        Returns:
            selected_task_tuples: List[Tuple[float, str]] - List of task tuples (pass rate, task path)
            remaining_slots: int - Number of slots remaining after selecting tasks
        """
        selected_task_tuples = []
        remaining_slots = num_to_adapt

        discard_impossible_tasks = self.cfg.task_generation.get(
            "discard_impossible_tasks", True
        )

        if discard_impossible_tasks:
            self.logger.info("Discarding impossible tasks.")
            all_tasks = hard_tasks + easy_tasks + medium_tasks
        else:
            all_tasks = (
                impossible_tasks + hard_tasks + easy_tasks + medium_tasks
            )

        if all_tasks:
            num_samples = num_to_adapt
            # selected_task_tuples = self.random_instance.sample(
            #     all_tasks, num_samples
            # )
            selected_task_tuples = self.random_instance.choices(
                all_tasks, k=num_samples
            )
            remaining_slots = num_to_adapt - num_samples
        else:
            if discard_impossible_tasks:
                self.logger.warning(
                    "No tasks available for uniform selection. Adding back impossible tasks."
                )
                all_tasks.extend(impossible_tasks)
                num_samples = num_to_adapt

                selected_task_tuples = self.random_instance.choices(
                    all_tasks, k=num_samples
                )
                remaining_slots = num_to_adapt - len(selected_task_tuples)
            else:
                self.logger.warning(
                    "No tasks available for uniform selection. Returning empty list."
                )

        return selected_task_tuples, remaining_slots

    def adapt_task_pool(
        self,
        archive_data: ACDCArchiveData,
        current_gen: int,
    ) -> List[str]:
        """
        Adapts the task pool based on model performance in the archive using multiprocessing.
        Orchestrates pass rate calculation, prioritization, generation, validation, and pruning.

        Args:
            archive_data: Dictionary containing the current DNS archive ('dns_archive').
            current_gen: The current generation number (for logging/naming).

        Returns:
            A list of paths to the newly generated and validated tasks added during this cycle.
        """
        self.logger.info(
            f"Starting task pool adaptation for generation {current_gen} using multiprocessing."
        )
        dns_archive: List[ACDCSolution] = archive_data.get("dns_archive", [])
        new_tasks_for_post_adapt_eval: List[str] = []
        successfully_replaced_originals: Dict[str, float] = {}

        if not dns_archive:
            self.logger.warning(
                "DNS archive is empty. Skipping task adaptation."
            )
            return []
        if not self.tasks:
            self.logger.warning(
                "Current task pool is empty. Skipping task adaptation."
            )
            return []

        # 1. Calculate Pass Rates
        (
            task_pass_rates,
            task_pass_counts,
            task_total_counts,
            current_task_ids_paths,
        ) = self._calculate_task_pass_rates(dns_archive)

        # 2. Prioritize Tasks (returns categorized lists)
        impossible_tasks, hard_tasks, easy_tasks, medium_tasks = (
            self._prioritize_tasks_for_adaptation(
                task_pass_rates, current_task_ids_paths
            )
        )

        # Step 2 (New): Handle existing "impossible" children with parents in active limbo
        if (
            self.enable_conditional_parent_replacement
            and self.cfg.task_generation.discard_impossible_tasks
        ):
            # Work with a copy of impossible_tasks to avoid modification during iteration
            impossible_tasks_copy = impossible_tasks.copy()
            for pass_rate, current_child_path in impossible_tasks_copy:
                # Check if this impossible task is a child whose parent is in active limbo
                if current_child_path in self.active_limbo_map:
                    # Retrieve and remove the parent path from active limbo
                    original_parent_path = self.active_limbo_map.pop(
                        current_child_path
                    )

                    # Remove the impossible child from the pool
                    if current_child_path in self.tasks:
                        self.remove_task_and_vector_db(
                            task_path=current_child_path,
                            vector_dbs=[self.vector_db_active],
                        )

                    # Reinstate the parent if not already in the pool
                    if original_parent_path not in self.tasks:
                        task_code, relevant_metadata, task_id, _ = (
                            self._load_task_code_and_metadata(
                                original_parent_path
                            )
                        )
                        self.add_task_and_vector_db(
                            task_path=original_parent_path,
                            code=task_code,
                            metadata=relevant_metadata,
                            custom_id=task_id,
                            vector_dbs=[self.vector_db_active],
                        )
                        # Add as newly generated task so that post-adapt eval occurs for the current model archive
                        new_tasks_for_post_adapt_eval.append(
                            original_parent_path
                        )

                    self.logger.info(
                        f"Existing child task {os.path.basename(current_child_path)} found impossible. "
                        f"Reinstating parent {os.path.basename(original_parent_path)} from active limbo. "
                        f"Removing child {os.path.basename(current_child_path)}."
                    )

                    # Remove from the categorized lists to prevent re-selection in this cycle
                    # Find and remove from all original lists (the non-copy versions)
                    impossible_tasks = [
                        t
                        for t in impossible_tasks
                        if t[1] != current_child_path
                    ]
                    hard_tasks = [
                        t for t in hard_tasks if t[1] != current_child_path
                    ]
                    easy_tasks = [
                        t for t in easy_tasks if t[1] != current_child_path
                    ]
                    medium_tasks = [
                        t for t in medium_tasks if t[1] != current_child_path
                    ]

        # Remove impossible tasks from task pool if option is set
        if self.cfg.task_generation.discard_impossible_tasks and len(
            impossible_tasks
        ) < len(self.tasks):
            for _, task_path in impossible_tasks:
                if task_path in self.tasks:
                    self.remove_task_and_vector_db(
                        task_path=task_path, vector_dbs=[self.vector_db_active]
                    )

        # 3. Select Tasks for Adaptation (New Logic)
        num_to_adapt = self.cfg.acdc.get(
            "new_tasks_per_generation", 25
        )  # Use configured value

        if self.cfg.task_generation.get(
            "use_biased_parent_task_selection", True
        ):
            self.logger.info("Using difficulty-biased parent task selection.")
            selected_task_tuples, remaining_slots = (
                self._get_tasks_for_adaptation_biased(
                    impossible_tasks=impossible_tasks,
                    hard_tasks=hard_tasks,
                    easy_tasks=easy_tasks,
                    medium_tasks=medium_tasks,
                    num_to_adapt=num_to_adapt,
                )
            )
        else:
            self.logger.info("Using uniform parent task selection.")
            selected_task_tuples, remaining_slots = (
                self._get_tasks_for_adaptation_uniform(
                    impossible_tasks=impossible_tasks,
                    hard_tasks=hard_tasks,
                    easy_tasks=easy_tasks,
                    medium_tasks=medium_tasks,
                    num_to_adapt=num_to_adapt,
                )
            )

        self.logger.info(
            f"Selected a total of {len(selected_task_tuples)} tasks for adaptation. ({num_to_adapt - remaining_slots} selected / {num_to_adapt} requested)"
        )

        if not selected_task_tuples:
            self.logger.info("No tasks selected for adaptation in this cycle.")
            return []

        # 4. Prepare Arguments for Parallel Generation
        num_workers = self.cfg.acdc.get(
            "num_adaptation_workers", multiprocessing.cpu_count()
        )
        self.logger.info(f"Using {num_workers} workers for adaptation.")

        # Prepare arguments for the adaptation worker, including the calculated task_number
        adapt_args = []
        for adapt_idx, (original_pass_rate, original_task_path) in enumerate(
            selected_task_tuples
        ):
            # Determine adaptation type and prompt here before passing to worker
            adapt_type, prompt_template = (
                self._determine_adaptation_type_and_prompt(
                    original_pass_rate, original_task_path
                )
            )
            if not adapt_type or not prompt_template:
                self.logger.warning(
                    f"Could not determine adaptation type/prompt for {os.path.basename(original_task_path)}, skipping adaptation."
                )
                continue  # Skip this task if type/prompt determination fails

            # Use and increment the global counter for the task number
            self.task_counter += 1
            task_number = self.task_counter

            # Step 4 (New): Move selected parent to pending limbo for "easier" or "harder" adaptation
            if self.enable_conditional_parent_replacement and adapt_type in [
                "easier",
                "harder",
            ]:
                # Remove the parent from the task pool
                if original_task_path in self.tasks:
                    self.remove_task_and_vector_db(
                        task_path=original_task_path,
                        vector_dbs=[self.vector_db_active],
                    )

                # Store the parent in pending limbo
                self.pending_limbo_parents[task_number] = original_task_path

                self.logger.info(
                    f"Moved parent {os.path.basename(original_task_path)} to pending limbo "
                    f"for new child generation attempt {task_number}."
                )

            adapt_args.append(
                (
                    original_task_path,
                    original_pass_rate,
                    task_number,  # Pass calculated task number
                    adapt_type,  # Pass determined adaptation type
                    prompt_template,  # Pass determined prompt template
                )
            )

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(self._adapt_task_worker, adapt_args)

        # Step 6 (New): Process results with conditional parent replacement logic
        for result_tuple in results:
            (
                new_task_path,
                original_task_path,
                original_pass_rate,
                child_task_number,
            ) = result_tuple

            # Handle pending limbo logic if feature is enabled
            if (
                self.enable_conditional_parent_replacement
                and child_task_number in self.pending_limbo_parents
            ):
                original_parent_path = self.pending_limbo_parents.pop(
                    child_task_number
                )

                if new_task_path:  # Child generation succeeded
                    # Add child to newly generated tasks
                    new_tasks_for_post_adapt_eval.append(new_task_path)
                    # Store parent in active limbo, linked to successful child
                    self.active_limbo_map[new_task_path] = original_parent_path

                    self.logger.info(
                        f"New child {os.path.basename(new_task_path)} successfully generated from parent "
                        f"{os.path.basename(original_parent_path)}. Parent {os.path.basename(original_parent_path)} "
                        f"moved to active limbo, linked to child {os.path.basename(new_task_path)}."
                    )
                else:  # Child generation failed
                    # Reinstate parent to task pool
                    if original_parent_path not in self.tasks:
                        task_code, relevant_metadata, task_id, _ = (
                            self._load_task_code_and_metadata(
                                original_parent_path
                            )
                        )
                        self.add_task_and_vector_db(
                            task_path=original_parent_path,
                            code=task_code,
                            metadata=relevant_metadata,
                            custom_id=task_id,
                            vector_dbs=[self.vector_db_active],
                        )

                    self.logger.info(
                        f"Initial child generation from parent {os.path.basename(original_parent_path)} failed. "
                        f"Reinstating parent {os.path.basename(original_parent_path)} from pending limbo."
                    )
            else:  # Standard logic without conditional parent replacement
                if new_task_path:
                    new_tasks_for_post_adapt_eval.append(new_task_path)
                    # Record which original task led to a successful new task
                    successfully_replaced_originals[original_task_path] = (
                        original_pass_rate
                    )

        self.logger.info(
            f"Generated and validated {len(new_tasks_for_post_adapt_eval)} new tasks via adaptation."
        )

        # 5. Update Task Pool (Add new, remove replaced impossible)
        _, other_originals_replaced = self._update_pool_after_adaptation(
            new_tasks_for_post_adapt_eval, successfully_replaced_originals
        )

        # 6. Prune Task Pool if Needed (pass counts/totals no longer needed for pruning)
        self._prune_task_pool_if_needed(other_originals_replaced)

        # 7. Return Newly Added Tasks
        self.logger.info(
            f"Task pool adaptation finished. Final pool size: {len(self.tasks)}"
        )

        # Check current task pool to see if impossible tasks remain
        if self.cfg.task_generation.discard_impossible_tasks:
            for task_path, pass_rate in self.tasks_pass_rates.items():
                if pass_rate == 0.0:
                    assert (
                        task_path not in self.tasks
                    ), "Impossible task should have been removed, but still present"

        return new_tasks_for_post_adapt_eval

    def add_task_and_vector_db(
        self,
        task_path: str,
        code: str,
        metadata: dict,
        custom_id: Optional[str] = None,
        vector_dbs: Optional[list] = None,
        exclude_from_archive: bool = False,
    ):
        """
        Add a task to self.tasks and to the specified vector DBs (default: both active and historical if present).
        Args:
            task_path: Path to the task directory.
            code: Task code (for vector DB content).
            metadata: Metadata dict for the task.
            custom_id: Custom ID for the vector DB (usually the task dir basename).
            vector_dbs: List of vector DBs to add to (default: both active and historical if present).
            exclude_from_archive: If True, add to vector DBs but not to self.tasks (for seed tasks).
        """
        if custom_id is None:
            custom_id = os.path.basename(task_path)

        # Check if this is a seed task that should be excluded from the final archive
        task_id = metadata.get("task_id", custom_id)
        is_seed_task = (
            hasattr(self, "seed_task_ids") and task_id in self.seed_task_ids
        )

        # Add to self.tasks only if not excluding from archive and not a seed task
        if (
            not exclude_from_archive
            and not is_seed_task
            and task_path not in self.tasks
        ):
            self.tasks.append(task_path)
        elif is_seed_task:
            self.logger.debug(
                f"Excluding seed task {task_id} from final archive"
            )

        if vector_dbs is None:
            vector_dbs = []
            if self.vector_db_active is not None:
                vector_dbs.append(self.vector_db_active)
            if self.vector_db_historical is not None:
                vector_dbs.append(self.vector_db_historical)
        for db in vector_dbs:
            if db is not None:
                db.add_sample(content=code, metadata=metadata, custom_id=custom_id)

    def remove_task_and_vector_db(
        self, task_path: str, vector_dbs: Optional[list] = None
    ):
        """
        Remove a task from self.tasks and from the specified vector DBs (default: both active and historical if present).
        Args:
            task_path: Path to the task directory.
            vector_dbs: List of vector DBs to remove from (default: both active and historical if present).
        """
        if task_path in self.tasks:
            self.tasks.remove(task_path)
        custom_id = os.path.basename(task_path)
        if vector_dbs is None:
            vector_dbs = []
            if self.vector_db_active is not None:
                vector_dbs.append(self.vector_db_active)
            if self.vector_db_historical is not None:
                vector_dbs.append(self.vector_db_historical)
        for db in vector_dbs:
            if db is not None:
                db.delete_sample(custom_id)

    def _load_task_code_and_metadata(self, task_path: str):
        """
        Given a task_path, load the task code, relevant metadata, and task_id (basename).
        Returns:
            (task_code: str, relevant_metadata: dict, task_id: str)
        Raises:
            FileNotFoundError if task.json or task.py is missing.
        """
        task_id = os.path.basename(task_path)
        task_json_file = os.path.join(task_path, "task.json")
        task_py_file = os.path.join(task_path, "task.py")
        if not (
            os.path.exists(task_json_file) and os.path.exists(task_py_file)
        ):
            raise FileNotFoundError(f"Missing files for {task_path}")
        with open(task_json_file, "r") as f:
            task_json = json.load(f)
        with open(task_py_file, "r") as f:
            task_code = f.read()
        task_json["task_family"] = task_code
        relevant_metadata = {
            k: v
            for k, v in task_json.items()
            if k != "task_family" and k != "done"
        }
        return task_code, relevant_metadata, task_id, task_json
