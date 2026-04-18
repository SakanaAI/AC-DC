# Task Unit Tests

Unit tests for the task subsystem: vector database, task generation, and Docker sandbox execution.

## Test Files

### `test_simple_vectordb.py`
Tests for `SimpleVectorDB` — a lock-free, directory-based vector database for task embeddings.

- **TestVectorDBCore** (11 tests): add, get, delete, update, sanitization, counting; atomic file ops and cleanup on failure
- **TestVectorDBBatch** (2 tests): batch operations with success/failure handling
- **TestVectorDBExportImport** (2 tests): zip export and import with merge/overwrite strategies

### `test_vectordb_search.py`
Tests for similarity search and configuration persistence.

- **TestVectorDBSearch** (10 tests): similarity search by metadata/content, threshold filtering, top-N limiting, descending score ordering
- **TestVectorDBEmbedding** (2 tests): embedding generation and zero-vector fallback on error
- **TestVectorDBConfigPersistence** (2 tests): config save/load round-trip

### `test_task_generation.py`
Tests for `ACDCTaskPool` (`tasks/task_generation.py`). LLM, vector DB, and sandbox are mocked; file I/O uses `tmp_path`.

- **TestACDCTaskPoolInitialization**: pool construction, config validation, seed task loading
- **TestACDCTaskPoolAdaptation**: task adaptation logic, LLM prompting, deduplication via vector DB
- **TestACDCTaskPoolHelpers**: helper methods (task validation, path resolution, etc.)

### `test_docker_sandbox.py`
Integration-style tests for Docker sandbox execution (`tasks/docker_sandbox.py`). The LLM judge (vLLM server) is mocked via a local HTTP server; everything else runs against a real Docker daemon.

- **Timeout / resource limit** calculation helpers
- **Sandbox execution**: code runs, output capture, timeout enforcement
- **Judge integration**: mock vLLM server responses, retry and error handling

> **Requirements:** Docker daemon must be running and the `acdc-sandbox:latest` image must be built.
> Pass `--run-docker` to pytest to enable these tests:
> ```bash
> pytest tests/unit/test_tasks/test_docker_sandbox.py --run-docker
> ```

## Running the Tests

```bash
# All task tests
pytest tests/unit/test_tasks/ -v

# Individual files
pytest tests/unit/test_tasks/test_simple_vectordb.py -v
pytest tests/unit/test_tasks/test_vectordb_search.py -v
pytest tests/unit/test_tasks/test_task_generation.py -v

# Docker sandbox (requires Docker)
pytest tests/unit/test_tasks/test_docker_sandbox.py --run-docker -v

# With coverage
pytest tests/unit/test_tasks/ --cov=tasks --cov-report=html
```
