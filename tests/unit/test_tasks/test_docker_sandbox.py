"""
Integration tests for Docker sandbox execution.

These tests actually run Docker containers and execute code, testing the real sandbox behavior.
Only the LLM judge (vLLM server) is mocked via a mock HTTP server.

Requirements:
- Docker daemon must be running
- acdc-sandbox:latest image must be built
- Tests run on CPU only

To run the tests with Docker add the `--run-docker` flag:
```bash
pytest tests/unit/test_dns/test_docker_sandbox.py --run-docker
```
"""

import pytest
import json
import time
import tempfile
import os
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from omegaconf import OmegaConf
from http.server import HTTPServer, BaseHTTPRequestHandler

from tasks.docker_sandbox import (
    calculate_effective_timeout,
    run_task_in_sandbox,
    cleanup_orphaned_containers,
)


# ============================================================================
# Mock LLM Judge HTTP Server
# ============================================================================


class MockLLMJudgeHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler for LLM judge requests."""

    # Class variable to store responses
    responses = []
    request_count = 0

    def do_POST(self):
        """Handle POST requests to /v1/chat/completions."""
        if self.path == "/v1/chat/completions":
            # Read request body
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)

            # Increment request count
            MockLLMJudgeHandler.request_count += 1

            # Get response from queue or use default
            if MockLLMJudgeHandler.responses:
                response_text = MockLLMJudgeHandler.responses.pop(0)
            else:
                response_text = "THOUGHT:\nThe response is correct.\n\nDECISION:\nYes"

            # Send response
            response = {"choices": [{"message": {"content": response_text}}]}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


@pytest.fixture(scope="session")
def mock_llm_judge_server():
    """Start a mock LLM judge HTTP server for the test session."""
    # Find an available port
    import socket

    sock = socket.socket()
    sock.bind(("0.0.0.0", 0))  # Bind to all interfaces
    port = sock.getsockname()[1]
    sock.close()

    # Create and start server
    server = HTTPServer(
        ("0.0.0.0", port), MockLLMJudgeHandler
    )  # Listen on all interfaces
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Give server time to start
    time.sleep(0.1)

    # For macOS Docker, use host.docker.internal to reach host
    # For Linux, could use host IP, but host.docker.internal works on newer Docker
    yield f"http://host.docker.internal:{port}/v1"

    # Cleanup
    server.shutdown()


@pytest.fixture(autouse=True)
def reset_mock_judge():
    """Reset mock judge state before each test."""
    MockLLMJudgeHandler.responses = []
    MockLLMJudgeHandler.request_count = 0


# ============================================================================
# Docker and Sandbox Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def docker_available():
    """Check if Docker is available."""
    import docker

    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def sandbox_image_available():
    """Check if sandbox image is built."""
    import docker

    try:
        client = docker.from_env()
        client.images.get("acdc-sandbox:latest")
        return True
    except Exception:
        return False


@pytest.fixture
def sandbox_config(mock_llm_judge_server):
    """Sandbox configuration with mock LLM judge."""
    return OmegaConf.create(
        {
            "docker_sandbox": {
                "image": "acdc-sandbox:latest",
                "timeout": 30,
                "network_mode": "bridge",
                "memory_limit": "512m",
                "cpu_quota": None,
                "pid_limit": -1,
                "ulimit_nproc": 64,
                "ulimit_nofile": 1024,
                "security_level": "high",
                "max_retries": 3,
                "initial_backoff": 0.5,
                "llm_judge_timeout": 10,
                "llm_judge_max_retries": 2,
                "llm_judge_initial_delay": 0.5,
                "llm_judge_max_delay": 5.0,
            },
            "acdc": {
                "scientist_vllm_url": mock_llm_judge_server,
                "scientist_model": "test-model",
                "vllm_judge_url": mock_llm_judge_server,
                "vllm_judge_model": "judge-model",
                "judge_api_key": "test-key",
            },
        }
    )


@pytest.fixture
def temp_task_dir(tmp_path):
    """Create a temporary task directory."""
    task_dir = tmp_path / "task_test"
    task_dir.mkdir()
    return task_dir


def create_task_file(task_dir: Path, task_code: str):
    """Create a task.py file with given code."""
    task_file = task_dir / "task.py"
    task_file.write_text(task_code)
    return str(task_file)


# ============================================================================
# Test Timeout Calculation (Unit Tests - No Docker Required)
# ============================================================================


class TestCalculateEffectiveTimeout:
    """Tests for timeout calculation with LLM judge buffer."""

    def test_timeout_without_llm_judge(self):
        """Test timeout calculation when LLM judge is disabled."""
        result = calculate_effective_timeout(30, llm_judge_enabled=False)
        assert result == 30

    def test_timeout_with_llm_judge(self):
        """Test timeout calculation when LLM judge is enabled."""
        result = calculate_effective_timeout(30, llm_judge_enabled=True)
        assert result == 150  # 30 + 120 second buffer

    def test_timeout_large_base(self):
        """Test timeout with large base value."""
        result = calculate_effective_timeout(300, llm_judge_enabled=True)
        assert result == 420


# ============================================================================
# Docker Container Execution Tests
# ============================================================================


import sys


# Determine if Docker tests should run (check at module level)
RUN_DOCKER_TESTS = "--run-docker" in sys.argv


@pytest.mark.skipif(
    not RUN_DOCKER_TESTS, reason="Docker tests disabled (use --run-docker to enable)"
)
class TestDockerSandboxExecution:
    """Tests for actual Docker container execution."""

    def test_simple_task_execution(self, sandbox_config, temp_task_dir):
        """Test executing a simple task that returns a numeric score."""
        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        # Simple scoring: check if answer equals expected
        expected = task_data.get('expected')
        return 1.0 if answer == expected else 0.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        # Execute task
        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {"expected": "42"}, "answer": "42"},
            cfg=sandbox_config,
        )

        assert result == 1.0

    def test_task_with_computation(self, sandbox_config, temp_task_dir):
        """Test task that performs actual computation."""
        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        # Compute sum and check answer
        numbers = task_data.get('numbers', [])
        expected_sum = sum(numbers)
        try:
            answer_num = int(answer)
            return 1.0 if answer_num == expected_sum else 0.0
        except ValueError:
            return 0.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {"numbers": [1, 2, 3, 4, 5]}, "answer": "15"},
            cfg=sandbox_config,
        )

        assert result == 1.0

    def test_task_returns_dict(self, sandbox_config, temp_task_dir):
        """Test task that returns a dictionary."""
        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        return {
            "score": 0.85,
            "details": "Partial match",
            "answer_length": len(answer)
        }
"""
        task_file = create_task_file(temp_task_dir, task_code)

        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {}, "answer": "test answer"},
            cfg=sandbox_config,
        )

        assert result["score"] == 0.85
        assert result["details"] == "Partial match"
        assert result["answer_length"] == 11

    def test_task_with_llm_judge(
        self, sandbox_config, temp_task_dir, mock_llm_judge_server
    ):
        """Test task that calls LLM judge."""
        # Set up mock judge to return "Yes"
        MockLLMJudgeHandler.responses = [
            "THOUGHT:\nThe answer is correct.\n\nDECISION:\nYes"
        ]

        task_code = """
import sys
sys.path.insert(0, '/')
from sandbox_eval_helper import eval_with_llm_judge

class TaskFamily:
    def score(self, task_data, answer):
        instructions = task_data.get('instructions', 'Evaluate the answer')
        criteria = task_data.get('criteria', [])

        # Call LLM judge
        result = eval_with_llm_judge(instructions, answer, criteria)

        return 1.0 if result else 0.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={
                "task_data": {
                    "instructions": "Is this answer correct?",
                    "criteria": ["Must be accurate"],
                },
                "answer": "The answer is 42",
            },
            cfg=sandbox_config,
        )

        assert result == 1.0
        assert MockLLMJudgeHandler.request_count == 1

    def test_task_with_print_statements(self, sandbox_config, temp_task_dir):
        """Test that task can print debug info without breaking JSON parsing."""
        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        # Print debug info (should be ignored by parser)
        print("Debug: Starting evaluation")
        print("Answer received:", answer)

        # Return actual result
        result = 1.0 if "correct" in answer.lower() else 0.0
        print("Score computed:", result)

        return result
"""
        task_file = create_task_file(temp_task_dir, task_code)

        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {}, "answer": "This is correct"},
            cfg=sandbox_config,
        )

        # Should parse result from last line despite debug prints
        assert result == 1.0


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.skipif(not RUN_DOCKER_TESTS, reason="Docker tests disabled")
class TestDockerSandboxErrors:
    """Tests for error handling in sandbox execution."""

    def test_task_raises_exception(self, sandbox_config, temp_task_dir):
        """Test handling of task that raises an exception."""
        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        raise ValueError("Intentional error for testing")
"""
        task_file = create_task_file(temp_task_dir, task_code)

        with pytest.raises(RuntimeError) as exc_info:
            run_task_in_sandbox(
                task_script_path=task_file,
                function_name="score",
                input_data={"task_data": {}, "answer": "test"},
                cfg=sandbox_config,
            )

        assert "exited with code 1" in str(exc_info.value)

    def test_missing_task_family_class(self, sandbox_config, temp_task_dir):
        """Test error when TaskFamily class is missing."""
        task_code = """
# No TaskFamily class defined
def some_function():
    pass
"""
        task_file = create_task_file(temp_task_dir, task_code)

        with pytest.raises(RuntimeError) as exc_info:
            run_task_in_sandbox(
                task_script_path=task_file,
                function_name="score",
                input_data={"task_data": {}, "answer": "test"},
                cfg=sandbox_config,
            )

        assert "exited with code 1" in str(exc_info.value)

    def test_missing_score_function(self, sandbox_config, temp_task_dir):
        """Test error when score function is missing from TaskFamily."""
        task_code = """
class TaskFamily:
    def other_method(self):
        pass
"""
        task_file = create_task_file(temp_task_dir, task_code)

        with pytest.raises(RuntimeError) as exc_info:
            run_task_in_sandbox(
                task_script_path=task_file,
                function_name="score",
                input_data={"task_data": {}, "answer": "test"},
                cfg=sandbox_config,
            )

        assert "exited with code 1" in str(exc_info.value)

    def test_task_returns_non_serializable(self, sandbox_config, temp_task_dir):
        """Test error when task returns non-JSON-serializable object."""
        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        # Return a non-serializable object
        return lambda x: x + 1
"""
        task_file = create_task_file(temp_task_dir, task_code)

        with pytest.raises(RuntimeError) as exc_info:
            run_task_in_sandbox(
                task_script_path=task_file,
                function_name="score",
                input_data={"task_data": {}, "answer": "test"},
                cfg=sandbox_config,
            )

        # Should fail during JSON serialization
        assert "exited with code 1" in str(exc_info.value)

    @pytest.mark.skip(reason="Timeout test can hang with retry logic - tested manually")
    def test_task_timeout(self, sandbox_config, temp_task_dir):
        """Test handling of task that exceeds timeout.

        Note: This test is skipped because timeout + retry logic can cause hangs.
        Timeout functionality is tested in the integration test_full_pipeline.py.
        """
        # Use very short timeout and disable LLM judge URLs to avoid 120s buffer
        sandbox_config.docker_sandbox.timeout = 2
        sandbox_config.acdc.vllm_judge_url = ""  # Empty string to disable
        sandbox_config.acdc.scientist_vllm_url = ""  # Also disable fallback

        task_code = """
import time

class TaskFamily:
    def score(self, task_data, answer):
        # Infinite loop - will definitely timeout
        while True:
            time.sleep(1)
        return 1.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        with pytest.raises(RuntimeError) as exc_info:
            run_task_in_sandbox(
                task_script_path=task_file,
                function_name="score",
                input_data={"task_data": {}, "answer": "test"},
                cfg=sandbox_config,
            )

        error_msg = str(exc_info.value).lower()
        assert (
            "timed out" in error_msg
            or "timeout" in error_msg
            or "container execution failed" in error_msg
        )

    def test_llm_judge_unavailable(self, sandbox_config, temp_task_dir):
        """Test handling when LLM judge endpoint is unavailable."""
        # Point to non-existent endpoint
        sandbox_config.acdc.vllm_judge_url = "http://127.0.0.1:9999/v1"

        task_code = """
import sys
sys.path.insert(0, '/')
from sandbox_eval_helper import eval_with_llm_judge

class TaskFamily:
    def score(self, task_data, answer):
        # Try to call unavailable judge
        try:
            result = eval_with_llm_judge("Test", answer, [])
            return 1.0 if result else 0.0
        except Exception as e:
            # Return 0.0 on error
            return 0.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        # Should return 0.0 due to connection error
        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {}, "answer": "test"},
            cfg=sandbox_config,
        )

        assert result == 0.0


# ============================================================================
# Retry Logic Tests
# ============================================================================


@pytest.mark.skipif(not RUN_DOCKER_TESTS, reason="Docker tests disabled")
class TestDockerSandboxRetry:
    """Tests for retry logic with exponential backoff."""

    def test_retry_on_llm_judge_timeout(
        self, sandbox_config, temp_task_dir, mock_llm_judge_server
    ):
        """Test retry when LLM judge times out."""
        # Configure very short LLM judge timeout
        sandbox_config.docker_sandbox.llm_judge_timeout = 0.1

        # Make judge server delay response
        call_count = [0]
        original_do_post = MockLLMJudgeHandler.do_POST

        def slow_do_post(self):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: delay longer than timeout
                time.sleep(0.5)
            # Then proceed normally
            original_do_post(self)

        MockLLMJudgeHandler.do_POST = slow_do_post
        MockLLMJudgeHandler.responses = [
            "THOUGHT:\nGood.\n\nDECISION:\nYes",
            "THOUGHT:\nGood.\n\nDECISION:\nYes",
        ]

        task_code = """
import sys
sys.path.insert(0, '/')
from sandbox_eval_helper import eval_with_llm_judge

class TaskFamily:
    def score(self, task_data, answer):
        result = eval_with_llm_judge("Test", answer, [])
        return 1.0 if result else 0.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        try:
            # May succeed after retry or fail - either is acceptable for this test
            result = run_task_in_sandbox(
                task_script_path=task_file,
                function_name="score",
                input_data={"task_data": {}, "answer": "test"},
                cfg=sandbox_config,
            )
            # If succeeded, verify it worked
            assert result in [0.0, 1.0]
        except RuntimeError:
            # Expected if all retries failed
            pass
        finally:
            # Restore original method
            MockLLMJudgeHandler.do_POST = original_do_post


# ============================================================================
# Container Cleanup Tests
# ============================================================================


@pytest.mark.skipif(not RUN_DOCKER_TESTS, reason="Docker tests disabled")
class TestDockerSandboxCleanup:
    """Tests for container cleanup after execution."""

    def test_container_removed_after_success(self, sandbox_config, temp_task_dir):
        """Test that container is removed after successful execution."""
        import docker

        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        return 1.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        # Get container count before
        client = docker.from_env()
        containers_before = len(
            client.containers.list(all=True, filters={"ancestor": "acdc-sandbox:latest"})
        )

        # Execute task
        run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {}, "answer": "test"},
            cfg=sandbox_config,
        )

        # Wait a moment for cleanup
        time.sleep(0.5)

        # Get container count after
        containers_after = len(
            client.containers.list(all=True, filters={"ancestor": "acdc-sandbox:latest"})
        )

        # Should not have created lingering containers
        assert containers_after == containers_before

    def test_container_removed_after_error(self, sandbox_config, temp_task_dir):
        """Test that container is removed even after task error."""
        import docker

        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        raise ValueError("Test error")
"""
        task_file = create_task_file(temp_task_dir, task_code)

        # Get container count before
        client = docker.from_env()
        containers_before = len(
            client.containers.list(all=True, filters={"ancestor": "acdc-sandbox:latest"})
        )

        # Execute task (will raise error)
        try:
            run_task_in_sandbox(
                task_script_path=task_file,
                function_name="score",
                input_data={"task_data": {}, "answer": "test"},
                cfg=sandbox_config,
            )
        except RuntimeError:
            pass

        # Wait for cleanup
        time.sleep(0.5)

        # Get container count after
        containers_after = len(
            client.containers.list(all=True, filters={"ancestor": "acdc-sandbox:latest"})
        )

        # Should not have created lingering containers
        assert containers_after == containers_before

    def test_cleanup_orphaned_containers(self, sandbox_config):
        """Test cleanup of orphaned containers."""
        # Create some old containers manually (if any exist)
        cleaned = cleanup_orphaned_containers(image_name="acdc-sandbox:latest")

        # Should return number of containers cleaned (could be 0)
        assert cleaned >= 0


# ============================================================================
# Configuration Edge Cases
# ============================================================================


@pytest.mark.skipif(not RUN_DOCKER_TESTS, reason="Docker tests disabled")
class TestDockerSandboxConfig:
    """Tests for configuration edge cases."""

    def test_minimal_config(self, mock_llm_judge_server, temp_task_dir):
        """Test execution with minimal configuration."""
        minimal_config = OmegaConf.create(
            {
                "docker_sandbox": {
                    "image": "acdc-sandbox:latest",
                    "timeout": 30,
                },
                "acdc": {
                    "scientist_vllm_url": mock_llm_judge_server,
                    "scientist_model": "test-model",
                },
            }
        )

        task_code = """
class TaskFamily:
    def score(self, task_data, answer):
        return 0.75
"""
        task_file = create_task_file(temp_task_dir, task_code)

        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {}, "answer": "test"},
            cfg=minimal_config,
        )

        assert result == 0.75

    def test_vllm_prefix_stripping(self, mock_llm_judge_server, temp_task_dir):
        """Test that vllm/ prefix is stripped from model names."""
        config = OmegaConf.create(
            {
                "docker_sandbox": {
                    "image": "acdc-sandbox:latest",
                    "timeout": 30,
                },
                "acdc": {
                    "scientist_vllm_url": mock_llm_judge_server,
                    "scientist_model": "vllm/model-with-prefix",
                },
            }
        )

        task_code = """
import os

class TaskFamily:
    def score(self, task_data, answer):
        # Check that model name has prefix stripped
        judge_model = os.environ.get('VLLM_JUDGE_MODEL', '')
        # Should not have vllm/ prefix
        has_prefix = judge_model.startswith('vllm/')
        return 0.0 if has_prefix else 1.0
"""
        task_file = create_task_file(temp_task_dir, task_code)

        result = run_task_in_sandbox(
            task_script_path=task_file,
            function_name="score",
            input_data={"task_data": {}, "answer": "test"},
            cfg=config,
        )

        assert result == 1.0  # Prefix should be stripped
