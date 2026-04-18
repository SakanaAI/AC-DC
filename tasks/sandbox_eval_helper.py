import os
import sys
import json
import re
import time
import random
import logging
from typing import Optional, List
from datetime import datetime

# Attempt to import openai, fail gracefully if not found (should be present in custom image)
try:
    from openai import OpenAI, APIError, APITimeoutError, APIConnectionError  # type: ignore
except ImportError:
    print(
        "Error: openai library not found. Cannot run LLM judge.",
        file=sys.stderr,
    )

    # Define dummy classes/exceptions if import fails to avoid NameErrors later
    # This allows the script to load but fail explicitly when called.
    class OpenAI:
        pass

    class APIError(Exception):
        pass

    class APITimeoutError(TimeoutError):
        pass

    class APIConnectionError(ConnectionError):
        pass

    # Set a flag or raise immediately? Let's let it fail during execution.


# Basic logger for sandbox execution
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Request tracking for monitoring
request_counter = 0
request_start_time = None


# --- Prompts (Copied from dns/task_gen_prompts.py) ---
judge_system_msg = """You are an impartial judge evaluating the quality of responses provided by an AI assistant.
You will be provided the instructions for the task and the assistant's submission in response.
You will decide whether or not the agent succeeded at the task.
Please consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
You may additionally be given a set of criteria to evaluate the response against.
Do not allow the length of the response to influence your decision.

Respond precisely in the following format:

THOUGHT:
<THOUGHT>

DECISION:
<DECISION>

In <THOUGHT>, briefly reason about the agent's response and whether it meets the additional criteria.

In <DECISION>, provide your answer as either "Yes" or "No"."""

judge_prompt = '''Instruction: {instructions}
Submission: {submission}

Additional Evaluation Criteria:
"""
{criteria}
"""
'''


# --- Helper Function ---
def extract_decision(response: str) -> Optional[str]:
    """Extracts the decision ('Yes' or 'No') from the judge's response."""
    decision_pattern = r"DECISION:\s*(Yes|No)\s*$"
    match = re.search(decision_pattern, response, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip().capitalize()
    logger.warning(
        f"Could not extract decision from judge response: {response}"
    )
    return None


# --- Helper function for retry with exponential backoff ---
def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays

    Returns:
        Result of the function if successful

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except (APITimeoutError, APIConnectionError, APIError) as e:
            last_exception = e
            if attempt < max_retries:
                # Add jitter if enabled
                if jitter:
                    jittered_delay = delay * (0.5 + 0.5 * random.random())
                else:
                    jittered_delay = delay

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {jittered_delay:.2f} seconds..."
                )
                time.sleep(jittered_delay)

                # Calculate next delay with exponential backoff
                delay = min(delay * exponential_base, max_delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed. Last error: {type(e).__name__}: {e}"
                )

    raise last_exception  # type: ignore


# --- Main Judge Function (To be called by TaskFamily.score) ---
def eval_with_llm_judge(
    instructions: str,
    submission: str,
    criteria: Optional[List[str]] = None,
) -> bool:
    """
    Evaluates a submission using an external vLLM judge via the OpenAI client.
    Reads connection details from environment variables.

    Includes retry logic with exponential backoff for handling transient failures.

    Args:
        instructions: The instructions for the task.
        submission: The submission to evaluate.
        criteria: Optional list of additional criteria strings.

    Returns:
        True if the judge decides "Yes", False otherwise (including errors).
    """
    global request_counter, request_start_time

    # Track request start
    request_counter += 1
    request_id = (
        f"req_{request_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    request_start_time = time.time()

    logger.info(
        f"[{request_id}] Starting LLM judge evaluation request #{request_counter}"
    )

    # Check if OpenAI client is available
    if not isinstance(OpenAI, type):  # Check if it's the actual class
        logger.error(
            f"[{request_id}] OpenAI library failed to import. Cannot call LLM judge."
        )
        return False

    # Read connection details from environment variables
    base_url = os.environ.get("VLLM_JUDGE_URL")
    model_name = os.environ.get("VLLM_JUDGE_MODEL")
    api_key = os.environ.get(
        "OPENAI_API_KEY", "dummy-key"
    )  # Default to dummy key

    # Get configurable timeout from environment (default 60s)
    timeout = float(os.environ.get("VLLM_JUDGE_TIMEOUT", "60.0"))

    # Get retry configuration from environment
    max_retries = int(os.environ.get("VLLM_JUDGE_MAX_RETRIES", "3"))
    initial_delay = float(os.environ.get("VLLM_JUDGE_INITIAL_DELAY", "1.0"))
    max_delay = float(os.environ.get("VLLM_JUDGE_MAX_DELAY", "60.0"))

    if not base_url:
        logger.error(
            f"[{request_id}] VLLM_JUDGE_URL environment variable not set."
        )
        return False
    if not model_name:
        logger.error(
            f"[{request_id}] VLLM_JUDGE_MODEL environment variable not set."
        )
        return False

    logger.info(
        f"[{request_id}] Contacting LLM judge: URL={base_url}, Model={model_name}, "
        f"Timeout={timeout}s, MaxRetries={max_retries}"
    )

    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    except Exception as e:
        logger.error(f"[{request_id}] Failed to initialize OpenAI client: {e}")
        return False

    criteria_str = (
        "\n".join([f"* {c}" for c in criteria]) if criteria else "None"
    )
    formatted_prompt = judge_prompt.format(
        instructions=instructions,
        submission=submission,
        criteria=criteria_str,
    )

    messages = [
        {"role": "system", "content": judge_system_msg},
        {"role": "user", "content": formatted_prompt},
    ]

    # Log truncated versions for debugging
    logger.debug(
        f"[{request_id}] Request details - Instructions length: {len(instructions)}, "
        f"Submission length: {len(submission)}, Criteria: {len(criteria) if criteria else 0}"
    )

    def make_request():
        """Inner function to make the actual API request."""
        logger.debug(f"[{request_id}] Sending request to LLM judge...")
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        return completion.choices[0].message.content

    try:
        # Use retry logic with exponential backoff
        response_text = retry_with_exponential_backoff(
            make_request,
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
        )

        # Calculate request duration
        request_duration = time.time() - request_start_time
        logger.info(
            f"[{request_id}] Received response from LLM judge after {request_duration:.2f}s. "
            f"Response preview: {response_text[:100]}..."
        )

    except APITimeoutError as e:
        request_duration = time.time() - request_start_time
        logger.error(
            f"[{request_id}] Request to LLM judge timed out after {request_duration:.2f}s "
            f"and {max_retries + 1} attempts. Error: {e}"
        )
        return False
    except APIConnectionError as e:
        request_duration = time.time() - request_start_time
        logger.error(
            f"[{request_id}] Could not connect to LLM judge server at {base_url} "
            f"after {request_duration:.2f}s and {max_retries + 1} attempts. Error: {e}"
        )
        return False
    except APIError as e:
        request_duration = time.time() - request_start_time
        logger.error(
            f"[{request_id}] LLM judge API error after {request_duration:.2f}s "
            f"and {max_retries + 1} attempts. "
            f"Status={getattr(e, 'status_code', 'N/A')}, Response={getattr(e, 'response', 'N/A')}"
        )
        return False
    except Exception as e:
        # Log detailed error for unexpected issues
        request_duration = time.time() - request_start_time
        logger.exception(
            f"[{request_id}] Unexpected error calling LLM judge after {request_duration:.2f}s: {e}"
        )
        return False

    # Extract decision
    decision = extract_decision(response_text)

    # Log final result
    logger.info(f"[{request_id}] LLM judge decision: {decision}")

    # Return True only if decision is explicitly "Yes"
    return decision == "Yes"


# --- Helper function to get the function name to callable mapping ---
def get_function_name_to_callable(string: str):
    """
    Executes a function string in a sandboxed environment.

    Args:
        func_string: A string containing one or multiple python function definitions.

    Returns:
        A dictionary mapping function names to their callables.
    """

    try:
        # Create namespace for the function
        namespace = {}

        # Extract the function strings from the string
        re_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(re_pattern, string, re.DOTALL)

        # Execute the function definitions
        for match in matches:
            exec(match, namespace)

        # Get the function name to callable mapping
        return {
            name: obj
            for name, obj in namespace.items()
            if callable(obj) and not name.startswith("__")
        }

    except Exception as e:
        raise e


def get_request_stats() -> dict:
    """
    Get statistics about LLM judge requests.

    Returns:
        Dictionary with request statistics
    """
    global request_counter
    return {
        "total_requests": request_counter,
        "timestamp": datetime.now().isoformat(),
    }


# Example self-test block (optional, won't run when imported)
if __name__ == "__main__":
    print("Sandbox Eval Helper Script")
    print("-" * 50)

    # Test extract_decision function
    print("\nTesting extract_decision function:")
    test_cases = [
        ("THOUGHT:\nThe response meets criteria.\nDECISION:\nYes", "Yes"),
        ("THOUGHT:\nResponse is lacking.\nDECISION: No", "No"),
        ("THOUGHT:\nBlah blah.\nDECISION: Maybe", None),
        ("DECISION: yes", "Yes"),
        ("DECISION: NO", "No"),
    ]

    for test_input, expected in test_cases:
        result = extract_decision(test_input)
        status = "✓" if result == expected else "✗"
        print(
            f"{status} Input: {test_input[:50]}... -> Got: {result}, Expected: {expected}"
        )

    # Show configuration from environment
    print("\nEnvironment Configuration:")
    env_vars = [
        ("VLLM_JUDGE_URL", "Not set"),
        ("VLLM_JUDGE_MODEL", "Not set"),
        ("VLLM_JUDGE_TIMEOUT", "60.0"),
        ("VLLM_JUDGE_MAX_RETRIES", "3"),
        ("VLLM_JUDGE_INITIAL_DELAY", "1.0"),
        ("VLLM_JUDGE_MAX_DELAY", "60.0"),
    ]

    for var, default in env_vars:
        value = os.environ.get(var, default)
        print(f"  {var}: {value}")

    # Show request stats
    print(f"\nRequest Statistics: {get_request_stats()}")

    print(
        "\nNote: Cannot test eval_with_llm_judge without env vars and running server"
    )
