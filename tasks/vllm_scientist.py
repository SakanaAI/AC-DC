import json
import logging
from typing import Tuple, List, Dict, Any, Optional, Union

from omegaconf import DictConfig
import backoff

# Use OpenAI client for vLLM interaction
from openai import (
    OpenAI,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Define exceptions for backoff (using OpenAI client exceptions)
RETRYABLE_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    # APIError might indicate server-side issues that could be temporary
    # Add specific status codes if needed, e.g., APIError with status 5xx
)


# Backoff strategy for network requests
def backoff_hdlr(details):
    logger.warning(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} with args {details['args']} and kwargs {details['kwargs']}"
    )


# Decorator needs to be defined before the function it wraps
def retry_with_backoff(max_tries=5):
    return backoff.on_exception(
        backoff.expo,
        RETRYABLE_EXCEPTIONS,
        max_tries=max_tries,
        on_backoff=backoff_hdlr,
    )


# Apply the decorator when defining the function
# Note: We get max_tries from config later, so we apply the decorator inside a wrapper or adjust dynamically.
# For simplicity now, let's apply it directly in the call within the test block,
# and assume the caller in task_generation will handle retries if needed, or apply decorator there.
# OR: Define the function without decorator and wrap it later. Let's do that.


def _get_vllm_response_core(
    prompt: Union[str, List[Dict[str, str]]],
    system_message: str,
    base_url: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    top_p: float = 0.95,
    api_key: str = "dummy-key",  # Required by client, but not used by local vLLM
    timeout: float = 60.0,  # Timeout for the API call itself
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Core logic to send a request to a vLLM OpenAI-compatible server using the openai client.
    (Retry logic should be applied by the caller if desired).
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    elif isinstance(prompt, list):
        messages = [
            {"role": "system", "content": system_message},
            *prompt,
        ]
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    extra_body = {}

    logger.debug(
        f"Sending request to vLLM OpenAI endpoint: BaseURL={base_url}, Model={model_name}"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body=(
                extra_body if extra_body else None
            ),  # Pass only if not empty
        )
        logger.info(f"Received response from vLLM: {completion}")

        if completion.choices:
            assistant_response = completion.choices[0].message.content
            if assistant_response is not None:
                # Append assistant response to history
                messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
                return assistant_response, messages
            else:
                logger.error("Assistant response content is None")
                raise ValueError("Assistant response content is None")
        else:
            logger.error(
                f"Invalid response format from vLLM (no choices): {completion}"
            )
            raise ValueError(
                "Invalid response format from vLLM server (no choices)"
            )

    except APITimeoutError:
        logger.error(f"Request to vLLM timed out after {timeout} seconds.")
        raise  # Re-raise for potential backoff by caller
    except APIConnectionError as e:
        logger.error(f"Could not connect to vLLM server at {base_url}: {e}")
        raise
    except APIError as e:
        logger.error(f"vLLM API error: {e}")
        raise
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while contacting vLLM via OpenAI client: {e}"
        )
        raise ValueError(
            f"An unexpected error occurred with OpenAI client: {e}"
        )


# Wrapper function to potentially add backoff later if needed globally
def get_vllm_response(
    prompt: Union[str, List[Dict[str, str]]],
    system_message: str,
    base_url: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    top_p: float = 0.95,
    api_key: str = "dummy-key",
    timeout: float = 60.0,
    # max_retries: int = 3 # If we want caller to specify retries
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Sends a request to a vLLM OpenAI-compatible server using the openai client.
    """
    # If applying backoff here:
    # decorated_func = retry_with_backoff(max_tries=max_retries)(_get_vllm_response_core)
    # return decorated_func(...)
    # For now, call core directly:
    return _get_vllm_response_core(
        prompt=prompt,
        system_message=system_message,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        api_key=api_key,
        timeout=timeout,
    )


def create_vllm_client_params(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """
    Creates a configuration dictionary for the vLLM client based on the main config.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        A dictionary containing vLLM client parameters, or None if disabled.
    """
    if not cfg.acdc.get("vllm_enabled", False):
        return None

    # Construct the full URL if not already fully specified
    base_url = cfg.acdc.get("scientist_vllm_url")
    if not base_url:
        host = cfg.acdc.get("vllm_server_host", "localhost")
        port = cfg.acdc.get("vllm_server_port", 8000)
        # Ensure v1 endpoint is included
        base_url = f"http://{host}:{port}/v1"
        # Check if already ends with /v1, handle potential double slashes if necessary
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"

    # Extract model name (strip 'vllm/' prefix if present, as it's just an identifier)
    model_identifier = cfg.acdc.get("scientist_model", "")
    # The actual model name used in the API call should be the one vLLM expects
    # Remove just the vllm/ prefix if present
    model_name = (
        model_identifier.split("/", 1)[-1]
        if "vllm/" in model_identifier
        else model_identifier
    )

    return {
        "base_url": base_url,
        "model_name": model_name,
        "temperature": cfg.acdc.get("vllm_temperature", 0.7),
        "max_tokens": cfg.acdc.get("vllm_max_tokens", 2000),
        "timeout": float(cfg.acdc.get("vllm_timeout", 60.0)),  # Ensure float
        "top_p": cfg.acdc.get("vllm_top_p", 0.95),
        "max_retries": cfg.acdc.get(
            "vllm_max_retries", 3
        ),  # Keep for potential use in caller
    }


# Example usage (for testing purposes)
if __name__ == "__main__":
    from omegaconf import OmegaConf  # Import here for testing block only

    # Mock config for testing
    mock_cfg = OmegaConf.create(
        {
            "acdc": {
                "vllm_enabled": True,
                "scientist_model": "vllm/Qwen/Qwen2.5-72B-Instruct",  # Identifier
                "vllm_server_host": "172.16.0.57",
                "vllm_server_port": 8001,
                # "scientist_vllm_url": "http://172.16.0.57:8001/v1", # Let create_vllm_client_params build it
                "vllm_temperature": 0.1,
                "vllm_max_tokens": 50,
                "vllm_timeout": 10,
                "vllm_top_p": 0.9,
                "vllm_max_retries": 3,
            }
        }
    )

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing vLLM client with OpenAI library...")

    client_params = create_vllm_client_params(mock_cfg)

    if client_params:
        logger.info(f"Client Params: {client_params}")
        test_prompt = "Write a short story about a brave kangaroo."
        test_system_message = "You are a helpful assistant."

        # Apply backoff dynamically for the test call
        get_vllm_response_test = retry_with_backoff(
            max_tries=client_params.get("max_retries", 3)
        )(_get_vllm_response_core)

        try:
            response_text, history = get_vllm_response_test(
                prompt=test_prompt,
                system_message=test_system_message,
                base_url=client_params["base_url"],
                model_name=client_params["model_name"],
                temperature=client_params["temperature"],
                max_tokens=client_params["max_tokens"],
                top_p=client_params["top_p"],
                timeout=client_params["timeout"],
            )
            logger.info(
                f"\n--- Test Response ---\n{response_text}\n--------------------"
            )
            logger.info(
                f"\n--- Message History ---\n{json.dumps(history, indent=2)}\n--------------------"
            )
        except Exception as e:
            logger.error(f"Test failed: {e}")
    else:
        logger.info("vLLM client params not created (check config).")
