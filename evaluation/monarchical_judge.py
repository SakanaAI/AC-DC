import os
import yaml
import logging
import time
import json
from typing import List, Type, Optional, Callable, Any
from functools import wraps
from datetime import datetime

from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)


# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Try to read the model url from the environment variable
MODEL_URL = os.getenv(
    "LLM_AS_A_JUDGE_MODEL_URL", "http://172.16.15.200:8001/v1"
)
MODEL_NAME = os.getenv("LLM_AS_A_JUDGE_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if MODEL_URL is None or MODEL_NAME is None:
    raise ValueError(
        "LLM_AS_A_JUDGE_MODEL_URL and LLM_AS_A_JUDGE_MODEL_NAME must be set"
    )
# check if the model url ends with /v1
if not MODEL_URL.endswith("/v1"):
    MODEL_URL = f"{MODEL_URL}/v1"


# Set up OpenAI client to locally hosted model
client = OpenAI(api_key="empty", base_url=MODEL_URL)

# Add a global logger for judge interactions
judge_logger = logging.getLogger("judge_interactions")
judge_logger.propagate = (
    False  # Prevent messages from going to parent loggers (terminal)
)
# Add a counter to track logged samples
_logged_samples_count = 0
MAX_SAMPLES_TO_LOG = 100


def setup_judge_logging(
    log_dir: str = ".logs/answer_selection_judge_interactions",
):
    """
    Set up logging for judge interactions.

    Args:
        log_dir: Directory to store judge interaction logs
    """
    global _logged_samples_count
    _logged_samples_count = 0  # Reset counter when setting up new logging

    # Create a file handler for judge interactions
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H_%M_%S")
    log_dir = os.path.join(log_dir, date_str)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"judge_interactions-{time_str}.jsonl")

    # Create a custom formatter for JSON logging
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            if hasattr(record, "judge_data"):
                return json.dumps(record.judge_data)
            return super().format(record)

    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    judge_logger.addHandler(file_handler)
    judge_logger.setLevel(logging.INFO)

    return log_file


def retry_on_specific_exceptions(
    on_exceptions: List[Type[Exception]],
    max_retries: Optional[int] = None,
    backoff_time: float = 3.0,
    backoff_multiplier: float = 1.5,
    on_exception_callback: Optional[Callable[[Exception, float], Any]] = None,
):
    """Retry on an LLM Provider's rate limit error with exponential backoff
    For example, to use for OpenAI, do the following:
    ```
    from openai import RateLimitError

    # Recommend specifying max_retries to avoid infinite loops!
    @retry_on_specific_exceptions([RateLimitError], max_retries=3)
    def completion(...):
        # Wrap OpenAI completion function here
        ...
    ```
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep_time = backoff_time
            attempt = 0
            while max_retries is None or attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except tuple(on_exceptions) as e:
                    if on_exception_callback is not None:
                        on_exception_callback(e, sleep_time)
                    time.sleep(sleep_time)
                    sleep_time *= backoff_multiplier
                    attempt += 1

        return wrapper

    return decorator


def _exception_callback(e: Exception, sleep_time: float) -> None:
    logger.warning(
        f"Request failed with error: {e}\nRetrying in {sleep_time} seconds"
    )


@retry_on_specific_exceptions(
    on_exceptions=[RateLimitError, APITimeoutError, APIConnectionError],
    max_retries=3,  # You can adjust this number
    backoff_time=1.0,  # Initial backoff time in seconds
    backoff_multiplier=2.0,  # Each retry will wait 2x longer
    on_exception_callback=_exception_callback,
)
def do_request(client: OpenAI, model_name: str, messages: list[dict]) -> dict:
    """
    Do a request to the OpenAI API.
    """
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        extra_headers={"X-Disable-Request-Logging": "true"},
    )


def ask_judge(
    messages: list[dict],
    extract_judge_decision: Callable[[str], int],
    context_info: Optional[dict] = None,
) -> tuple[int, str]:
    """
    Ask the judge to judge the submission.

    Args:
        messages: list[dict] - The messages to send to the judge.
        extract_judge_decision: Callable[[str], int] - A function to extract the judge's decision from the response.
            This function should return the index of the selected model.
        context_info: Optional[dict] - Additional context information for logging (e.g., selection_method, sample_id, etc.)

    Returns:
        - selected_model_idx: int - The index of the selected model.
        - judge_full_response: str - The full response from the judge.
    """
    global _logged_samples_count

    start_time = time.time()

    try:
        response = do_request(client, MODEL_NAME, messages)
        if not response:
            return -1, "No response from judge"
        judge_response = response.choices[0].message.content
        selected_model_idx = extract_judge_decision(judge_response)
        end_time = time.time()

        # Only log if we haven't reached the limit
        should_log = _logged_samples_count < MAX_SAMPLES_TO_LOG

        if should_log:
            # Prepare logging data
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": end_time - start_time,
                "model_name": MODEL_NAME,
                "model_url": MODEL_URL,
                "messages": messages,
                "judge_response": judge_response,
                "extracted_decision": selected_model_idx,
                "success": True,
                "error": None,
            }

            # Add context information if provided
            if context_info:
                log_data["context"] = context_info

            # Log the interaction
            record = logging.LogRecord(
                name="judge_interactions",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None,
            )
            record.judge_data = log_data
            judge_logger.handle(record)

            _logged_samples_count += 1

        return selected_model_idx, judge_response

    except Exception as e:
        end_time = time.time()

        # Only log errors if we haven't reached the limit
        should_log = _logged_samples_count < MAX_SAMPLES_TO_LOG

        if should_log:
            # Prepare error logging data
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": end_time - start_time,
                "model_name": MODEL_NAME,
                "model_url": MODEL_URL,
                "messages": messages,
                "judge_response": None,
                "extracted_decision": None,
                "success": False,
                "error": str(e),
            }

            # Add context information if provided
            if context_info:
                log_data["context"] = context_info

            # Log the error
            record = logging.LogRecord(
                name="judge_interactions",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None,
            )
            record.judge_data = log_data
            judge_logger.handle(record)

            _logged_samples_count += 1

        logger.error(f"Error in ask_judge: {e}")
        return -1, f"Error in ask_judge: {e}"