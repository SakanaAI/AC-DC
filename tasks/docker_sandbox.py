import docker
import docker.errors
import docker.types
import logging
import os
import json
import tempfile
import time
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Track orphaned containers for cleanup
orphaned_containers = set()

# Helper script to run inside the container
def calculate_effective_timeout(base_timeout: int, llm_judge_enabled: bool = True) -> int:
    """
    Calculate effective timeout accounting for LLM judge retry delays.
    
    Args:
        base_timeout: Base timeout from config (in seconds)
        llm_judge_enabled: Whether LLM judge calls are expected
        
    Returns:
        Adjusted timeout in seconds
    """
    if not llm_judge_enabled:
        return base_timeout
    
    # Account for LLM judge retries and delays
    # Default: 3 retries with exponential backoff starting at 1s
    # Max delay is 60s, so worst case: initial + 1s + 2s + 4s + 60s = ~70s per judge call
    # Add buffer for multiple judge calls in a single task
    llm_judge_buffer = 120  # 2 minutes buffer for LLM judge operations
    
    return base_timeout + llm_judge_buffer


def monitor_container_health(container, timeout: int, check_interval: float = 1.0) -> Tuple[bool, Optional[str]]:
    """
    Monitor container health and resource usage during execution.
    
    Args:
        container: Docker container object
        timeout: Maximum time to wait (seconds)
        check_interval: How often to check container status (seconds)
        
    Returns:
        Tuple of (completed_successfully, error_message)
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check if container is still running
            container.reload()
            status = container.status
            
            if status == "exited":
                # Container finished execution
                return True, None
            elif status in ["dead", "removing", "removed"]:
                return False, f"Container entered unhealthy state: {status}"
            elif status == "oomkilled":
                return False, "Container killed due to out of memory"
            
            # Optional: Check container stats for resource issues
            # stats = container.stats(stream=False)
            # This can add overhead, so only enable if needed
            
            time.sleep(check_interval)
            
        except docker.errors.NotFound:
            # Container was removed (possibly completed and auto-removed)
            return True, None
        except Exception as e:
            logger.warning(f"Error checking container health: {e}")
            # Continue monitoring unless it's a critical error
            
    # Timeout reached
    return False, f"Container execution timed out after {timeout} seconds"


# Helper script to run inside the container
CONTAINER_SCRIPT = """
import importlib.util
import json
import sys
import os

def run_task_function(script_path, function_name, input_data_json):
    try:
        # Load the module dynamically
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
            print(f"Error: Could not load spec for module {script_path}", file=sys.stderr)
            sys.exit(1)
        task_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = task_module
        spec.loader.exec_module(task_module)

        # Check if TaskFamily exists and has the function
        if not hasattr(task_module, 'TaskFamily'):
            print(f"Error: TaskFamily class not found in {script_path}", file=sys.stderr)
            sys.exit(1)

        task_instance = task_module.TaskFamily()

        if not hasattr(task_instance, function_name):
            print(f"Error: Function '{function_name}' not found in TaskFamily instance", file=sys.stderr)
            sys.exit(1)

        # Deserialize input data
        input_data = json.loads(input_data_json)
        task_data = input_data.get('task_data')
        answer = input_data.get('answer')

        # Call the function
        # Assuming the function signature is score(task_data, answer)
        result = getattr(task_instance, function_name)(task_data, answer)

        # Serialize and print the result
        print(json.dumps(result))

    except Exception as e:
        print(f"Error during task execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <task_script_path> <function_name> <input_data_json>", file=sys.stderr)
        sys.exit(1)

    script_path_in_container = sys.argv[1]
    function_to_call = sys.argv[2]
    json_input = sys.argv[3]

    run_task_function(script_path_in_container, function_to_call, json_input)
"""


def run_task_in_sandbox(
    task_script_path: str,
    function_name: str,
    input_data: Dict[str, Any],
    cfg: DictConfig,
    retry_attempt: int = 0,
) -> Any:
    """
    Runs a specific function from a Python script within a Docker container sandbox.
    
    Includes retry logic, improved timeout handling, and container health monitoring.

    Args:
        task_script_path: Absolute path to the Python script containing the task logic (e.g., task.py).
        function_name: The name of the function to call within the script's TaskFamily class (e.g., 'score').
        input_data: A dictionary containing the necessary input for the function (e.g., {'task_data': ..., 'answer': ...}).
        cfg: Hydra configuration object, expected to contain cfg.docker_sandbox settings.
        retry_attempt: Current retry attempt number (for internal use).

    Returns:
        The result returned by the executed function.

    Raises:
        RuntimeError: If Docker execution fails or the task script encounters an error.
        docker.errors.DockerException: For Docker-related errors.
    """
    client = docker.from_env()
    sandbox_cfg = cfg.docker_sandbox
    
    # Get retry configuration
    max_retries = sandbox_cfg.get("max_retries", 3)
    initial_backoff = sandbox_cfg.get("initial_backoff", 1.0)
    
    # Check if this is a retry and if we've exceeded max retries
    if retry_attempt >= max_retries:
        raise RuntimeError(
            f"Maximum retries ({max_retries}) exceeded for task {task_script_path}. "
            f"Please check container logs and increase timeout if needed."
        )

    task_dir = os.path.abspath(os.path.dirname(task_script_path))
    task_filename = os.path.basename(task_script_path)
    container_workspace = "/workspace"
    script_path_in_container = os.path.join(container_workspace, task_filename)

    # Serialize input data
    input_data_json = json.dumps(input_data)

    # Create a temporary file for the container script
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp_script:
        tmp_script.write(CONTAINER_SCRIPT)
        host_script_path = tmp_script.name
    container_script_path = "/run_script.py"  # Path inside container

    # Prepare volumes
    host_helper_path = os.path.abspath("dns/sandbox_eval_helper.py")
    if not os.path.exists(host_helper_path):
        logger.error(f"Sandbox helper script not found at: {host_helper_path}")
        # Decide how to handle - raise error or proceed without judge? Raising is safer.
        raise FileNotFoundError(
            f"Required sandbox helper script not found: {host_helper_path}"
        )

    volumes = {
        task_dir: {"bind": container_workspace, "mode": "ro"},
        host_script_path: {"bind": container_script_path, "mode": "ro"},
        # Mount the sandbox eval helper script
        host_helper_path: {"bind": "/sandbox_eval_helper.py", "mode": "ro"},
    }

    # Prepare command
    command = [
        "python",
        container_script_path,
        script_path_in_container,
        function_name,
        input_data_json,
    ]

    container = None
    try:
        start_time = time.time()
        logger.debug(f"Starting Docker container for task: {task_script_path}")
        # Prepare environment variables for the container
        # Get judge details from config, defaulting to scientist details if judge-specific keys are missing
        judge_url = cfg.acdc.get(
            "vllm_judge_url", cfg.acdc.get("scientist_vllm_url")
        )
        # Default model name needs careful handling - strip vllm/ prefix if present
        default_model_name = cfg.acdc.get("scientist_model", "default_model")
        if isinstance(
            default_model_name, str
        ) and default_model_name.startswith("vllm/"):
            default_model_name = default_model_name.split("/", 1)[-1]
        judge_model = cfg.acdc.get("vllm_judge_model", default_model_name)
        judge_api_key = cfg.acdc.get(
            "judge_api_key", "dummy-key"
        )  # Default to dummy

        if not judge_url:
            logger.warning(
                "Judge URL (vllm_judge_url or scientist_vllm_url) not found in config. LLM Judge calls will fail."
            )
            # Set dummy value to avoid Docker error, but helper script will fail
            judge_url = "http://error-url-not-set"
        if not judge_model:
            logger.warning(
                "Judge model (vllm_judge_model or scientist_model) not found in config. LLM Judge calls will fail."
            )
            judge_model = "error-model-not-set"

        # Calculate effective timeout accounting for LLM judge delays
        llm_judge_enabled = bool(judge_url and judge_model)
        base_timeout = sandbox_cfg.timeout
        effective_timeout = calculate_effective_timeout(base_timeout, llm_judge_enabled)
        
        logger.info(
            f"Container timeout: base={base_timeout}s, effective={effective_timeout}s "
            f"(LLM judge {'enabled' if llm_judge_enabled else 'disabled'})"
        )
        
        environment_vars = {
            "VLLM_JUDGE_URL": judge_url,
            "VLLM_JUDGE_MODEL": judge_model,
            "OPENAI_API_KEY": judge_api_key,
            # Add timeout configuration for LLM judge
            "VLLM_JUDGE_TIMEOUT": str(sandbox_cfg.get("llm_judge_timeout", 60)),
            "VLLM_JUDGE_MAX_RETRIES": str(sandbox_cfg.get("llm_judge_max_retries", 3)),
            "VLLM_JUDGE_INITIAL_DELAY": str(sandbox_cfg.get("llm_judge_initial_delay", 1.0)),
            "VLLM_JUDGE_MAX_DELAY": str(sandbox_cfg.get("llm_judge_max_delay", 60.0)),
            # Pass proxy vars if they exist on the host, as they might be needed inside container
            "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
            "NO_PROXY": os.environ.get("NO_PROXY", ""),
        }
        # Filter out empty proxy vars to avoid passing empty strings
        environment_vars = {k: v for k, v in environment_vars.items() if v}
        logger.debug(
            f"Passing environment variables to container: {environment_vars}"
        )

        container = client.containers.run(
            image=sandbox_cfg.image,
            command=command,
            volumes=volumes,
            environment=environment_vars,  # Pass environment variables
            working_dir="/",  # Or container_workspace if needed
            detach=True,
            network_mode=sandbox_cfg.get(
                "network_mode", "bridge"
            ),  # Allow network if needed for judge
            mem_limit=sandbox_cfg.get("memory_limit", "512m"),
            cpu_quota=sandbox_cfg.get("cpu_quota", None),  # Use None if not set
            # Add other resource limits from config if needed
            pids_limit=sandbox_cfg.get(
                "pid_limit", -1
            ),  # Default to -1 (no limit)
            ulimits=(
                [
                    docker.types.Ulimit(
                        name="nproc",
                        soft=sandbox_cfg.get("ulimit_nproc", 64),
                        hard=sandbox_cfg.get("ulimit_nproc", 64),
                    ),
                    docker.types.Ulimit(
                        name="nofile",
                        soft=sandbox_cfg.get("ulimit_nofile", 1024),
                        hard=sandbox_cfg.get("ulimit_nofile", 1024),
                    ),
                    # docker.types.Ulimit(name='cpu', soft=sandbox_cfg.get('ulimit_cpu', -1), hard=sandbox_cfg.get('ulimit_cpu', -1)), # May cause issues
                ]
                if sandbox_cfg.get("security_level", "high") != "low"
                else None
            ),  # Apply ulimits unless security is low
        )

        # Monitor container with health checks and improved timeout
        completed, error_msg = monitor_container_health(container, effective_timeout)
        
        if not completed:
            # Container didn't complete successfully
            try:
                # Try to stop the container gracefully
                container.stop(timeout=5)
                logger.warning(f"Container stopped due to: {error_msg}")
            except Exception as e:
                logger.warning(f"Failed to stop container gracefully: {e}")
                try:
                    container.kill()
                    logger.warning("Container killed forcefully")
                except Exception as kill_err:
                    logger.error(f"Failed to kill container: {kill_err}")
            
            # Check if we should retry
            if retry_attempt < max_retries - 1:
                backoff_time = initial_backoff * (2 ** retry_attempt)
                logger.info(
                    f"Retrying task {task_filename} after {backoff_time}s delay "
                    f"(attempt {retry_attempt + 2}/{max_retries})"
                )
                time.sleep(backoff_time)
                return run_task_in_sandbox(
                    task_script_path, function_name, input_data, cfg, retry_attempt + 1
                )
            else:
                raise RuntimeError(
                    f"Container execution failed: {error_msg}. Task: {task_filename}"
                )
        
        # Get container exit status
        try:
            result = container.wait(timeout=1)  # Should return immediately since container exited
            exit_code = result.get("StatusCode", -1)
        except Exception as e:
            logger.warning(f"Failed to get container exit code: {e}")
            exit_code = -1
            
        elapsed_time = time.time() - start_time

        stdout = (
            container.logs(stdout=True, stderr=False).decode("utf-8").strip()
        )
        stderr = (
            container.logs(stdout=False, stderr=True).decode("utf-8").strip()
        )

        logger.debug(
            f"Container finished in {elapsed_time:.2f}s. Exit code: {exit_code}"
        )
        if stderr:
            logger.warning(f"Container stderr for {task_filename}:\n{stderr}")

        if exit_code != 0:
            error_msg = f"Docker container for {task_filename} exited with code {exit_code}. Stderr: {stderr}"
            
            # Check if we should retry based on the error
            if retry_attempt < max_retries - 1:
                # Determine if error is retryable (e.g., transient network issues)
                retryable_errors = [
                    "connection refused",
                    "timeout",
                    "temporary failure",
                    "LLM judge timed out",
                    "Could not connect to LLM judge",
                ]
                
                if any(err in stderr.lower() for err in retryable_errors):
                    backoff_time = initial_backoff * (2 ** retry_attempt)
                    logger.info(
                        f"Retrying task {task_filename} due to retryable error after {backoff_time}s delay "
                        f"(attempt {retry_attempt + 2}/{max_retries})"
                    )
                    time.sleep(backoff_time)
                    return run_task_in_sandbox(
                        task_script_path, function_name, input_data, cfg, retry_attempt + 1
                    )
            
            raise RuntimeError(error_msg)

        # Deserialize result from stdout - take only the last line which should be the JSON result
        # because sometimes the scientists adds prints that we don't want
        try:
            # Split by newlines and take the last non-empty line
            stdout_lines = [
                line.strip() for line in stdout.split("\n") if line.strip()
            ]
            if not stdout_lines:
                raise RuntimeError(
                    f"No output received from container for {task_filename}"
                )

            last_line = stdout_lines[-1]
            task_result = json.loads(last_line)
            logger.debug(f"Task {task_filename} result: {task_result}")
            return task_result
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to decode JSON result from container stdout for {task_filename}. Stdout: {stdout}"
            )

    except docker.errors.ContainerError as e:
        logger.error(f"Docker container error for {task_filename}: {e}")
        if container:
            try:
                logs = container.logs().decode('utf-8')
                logger.error(f"Container logs:\n{logs}")
            except Exception as log_err:
                logger.error(f"Failed to retrieve container logs: {log_err}")
        
        # Retry logic for container errors
        if retry_attempt < max_retries - 1:
            backoff_time = initial_backoff * (2 ** retry_attempt)
            logger.info(
                f"Retrying task {task_filename} after container error. Delay: {backoff_time}s "
                f"(attempt {retry_attempt + 2}/{max_retries})"
            )
            time.sleep(backoff_time)
            return run_task_in_sandbox(
                task_script_path, function_name, input_data, cfg, retry_attempt + 1
            )
        
        raise RuntimeError(
            f"Container execution failed for {task_filename} after {retry_attempt + 1} attempts"
        ) from e
    except Exception as e:
        logger.error(f"Error running task in sandbox for {task_filename}: {e}")
        raise
    finally:
        # Enhanced container cleanup
        if container:
            cleanup_start = time.time()
            try:
                # First try graceful removal
                container.remove(force=False)
                logger.debug(f"Removed container {container.short_id} gracefully")
            except docker.errors.NotFound:
                pass  # Container already removed
            except Exception as e:
                # Force removal if graceful fails
                try:
                    container.remove(force=True)
                    logger.debug(f"Force removed container {container.short_id}")
                except docker.errors.NotFound:
                    pass  # Container already removed
                except Exception as force_err:
                    logger.warning(
                        f"Failed to remove container {container.short_id} after "
                        f"{time.time() - cleanup_start:.2f}s: {force_err}. "
                        f"Container may be orphaned and require manual cleanup."
                    )
                    # Track orphaned container for later cleanup
                    orphaned_containers.add(container.id)
        # Clean up the temporary script file
        try:
            os.remove(host_script_path)
        except OSError as e:
            logger.warning(
                f"Failed to remove temporary script {host_script_path}: {e}"
            )


def cleanup_orphaned_containers(image_name: str = "acdc-sandbox:latest", force: bool = True):
    """
    Clean up any orphaned containers from the specified image.
    
    This function can be called periodically to ensure no containers are left running.
    
    Args:
        image_name: Docker image name to filter containers
        force: Whether to force remove running containers
        
    Returns:
        Number of containers cleaned up
    """
    global orphaned_containers
    
    try:
        client = docker.from_env()
        cleaned = 0
        
        # Find all containers using the sandbox image
        all_containers = client.containers.list(all=True, filters={"ancestor": image_name})
        
        for container in all_containers:
            try:
                # Check if container has been running for too long (e.g., > 10 minutes)
                created_time = container.attrs.get('Created', '')
                if created_time:
                    # Docker returns ISO format timestamp
                    from datetime import datetime, timezone
                    created = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                    age = (datetime.now(timezone.utc) - created).total_seconds()
                    
                    if age > 600:  # 10 minutes
                        logger.warning(f"Found stale container {container.short_id} (age: {age:.0f}s)")
                        container.remove(force=force)
                        cleaned += 1
                        orphaned_containers.discard(container.id)
                        
            except Exception as e:
                logger.warning(f"Failed to clean up container {container.short_id}: {e}")
        
        # Also try to clean up tracked orphaned containers
        for container_id in list(orphaned_containers):
            try:
                container = client.containers.get(container_id)
                container.remove(force=force)
                cleaned += 1
                orphaned_containers.remove(container_id)
            except docker.errors.NotFound:
                orphaned_containers.remove(container_id)
            except Exception as e:
                logger.warning(f"Failed to clean up orphaned container {container_id}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} orphaned containers")
            
        return cleaned
        
    except Exception as e:
        logger.error(f"Error during orphaned container cleanup: {e}")
        return 0
