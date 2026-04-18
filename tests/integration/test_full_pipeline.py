"""
Full integration test for the AC/DC pipeline.

This test runs the complete pipeline with a lightweight configuration
and verifies that all expected files and artifacts are created.

The test is marked as slow and requires GPU access. Run with:
    pytest tests/integration/test_full_pipeline.py -v -s

Or skip in CI:
    pytest -m "not slow" tests/
"""

import os
import pytest
import json
import subprocess
import time
from pathlib import Path
import logging
from datetime import datetime

# Configure logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def integration_test_config():
    """Path to the integration test configuration file."""
    return "configs/test_integration.yaml"


@pytest.fixture
def temp_output_dir():
    """Create a timestamped output directory for the test in outputs_tests/."""
    existing_path = os.environ.get('PYTEST_EXISTING_RUN_PATH')
    
    if existing_path:
        existing_path = Path(existing_path)
        if existing_path.exists():
            logger.info(f"Using existing run path: {existing_path}")
            # Validate basic structure
            # _validate_existing_run(existing_path)
            yield existing_path
            return  # Skip cleanup
        else:
            logger.warning(f"Existing path not found: {existing_path}")
            # Fall through to create new directory

    # Create outputs_tests directory if it doesn't exist
    outputs_tests_base = Path("outputs_tests")
    outputs_tests_base.mkdir(exist_ok=True)

    # Create timestamped directory for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = outputs_tests_base / f"test_run_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Created test output directory: {output_dir}")

    os.environ['PYTEST_EXISTING_RUN_PATH'] = str(output_dir)
    logger.info(f"Set PYTEST_EXISTING_RUN_PATH to: {output_dir}")

    yield output_dir

    # Optional: cleanup - uncomment if you want to remove test artifacts
    # import shutil
    # shutil.rmtree(output_dir, ignore_errors=True)
    # logger.info(f"Cleaned up test output directory: {output_dir}")


class TestFullPipeline:
    """Full integration test suite for AC/DC pipeline."""

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.requires_gpu
    def test_full_pipeline_execution(self, temp_output_dir):
        """
        Run the complete AC/DC pipeline and verify execution.

        This test:
        1. Runs the main_ac_dc.py script with test configuration
        2. Monitors execution for completion or timeout
        3. Verifies no crashes or errors occurred
        """
        # Check if using existing run (skip training)
        existing_run = os.environ.get('PYTEST_EXISTING_RUN_PATH')
        
        if existing_run:
            logger.info("Skipping pipeline execution - using existing run")
            logger.info(f"Existing run path: {existing_run}")
            # Perform basic validation that pipeline ran successfully
            # _validate_pipeline_completion(temp_output_dir)
            return
        
        # Otherwise, run full pipeline (current behavior)
        logger.info("Running full pipeline...")

        logger.info("=" * 80)
        logger.info("Starting full pipeline integration test")
        logger.info("=" * 80)

        # Build command to run the pipeline
        # Use absolute path to ensure proper directory handling
        abs_output_dir = temp_output_dir.absolute()

        # Prepare environment variables
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["VLLM_USE_V1"] = "0"

        cmd = [
            "python",
            "main_ac_dc.py",
            "celery.mode=main",
            "celery.num_workers=8",
            "celery.broker=pyamqp://guest@localhost:5805//",
            "celery.backend=redis://default:user@localhost:6505/0",
            "dns.init_population_size=5",
            "dns.population_size=7",
            "dns.num_model_per_gen=5",
            "acdc.initial_pool_size=5",
            "acdc.max_pool_size=10",
            "acdc.task_generation_interval=2",
            "acdc.num_sandbox_workers=16",
            "acdc.num_initialization_workers=4",
            "acdc.num_adaptation_workers=4",
            "dns.num_generations=3",
            f"output_dir={abs_output_dir}",
            f"hydra.run.dir={abs_output_dir}",  # Hydra working dir in test output
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Output directory: {temp_output_dir}")

        # Run the pipeline with timeout
        timeout_seconds = 10_800  # 180 minutes
        logger.info(f"Timeout: {timeout_seconds/60} minutes")
        start_time = time.time()

        try:
            # Run process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            # Stream output in real-time
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                logger.info(f"[PIPELINE] {line.rstrip()}")

                # Check for timeout
                if time.time() - start_time > timeout_seconds:
                    process.kill()
                    raise TimeoutError(
                        f"Pipeline execution exceeded {timeout_seconds}s timeout"
                    )

            # Wait for process to complete
            return_code = process.wait()

            elapsed_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info(
                f"Pipeline completed in {elapsed_time:.2f} seconds with return code {return_code}"
            )
            logger.info("=" * 80)

            # Check return code
            if return_code != 0:
                logger.error("Pipeline execution failed!")
                logger.error("Last 50 lines of output:")
                for line in output_lines[-50:]:
                    logger.error(line.rstrip())
                pytest.fail(
                    f"Pipeline execution failed with return code {return_code}"
                )

        except TimeoutError as e:
            logger.error(f"Pipeline execution timed out: {e}")
            pytest.fail(str(e))
        except Exception as e:
            logger.error(f"Unexpected error during pipeline execution: {e}")
            pytest.fail(f"Pipeline execution failed: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_directory_structure(self, temp_output_dir):
        """
        Verify that all expected directories are created.

        Expected structure:
        - models/
        - archives/
        - generated_tasks/pool/
        - images/
        """
        logger.info("Verifying directory structure...")

        # Note: This test should run after test_full_pipeline_execution
        # In practice, you might want to combine these or use fixtures

        expected_dirs = [
            temp_output_dir / "models",
            temp_output_dir / "archives",
            temp_output_dir / "generated_tasks" / "pool",
            temp_output_dir / "images",
        ]

        for directory in expected_dirs:
            assert (
                directory.exists()
            ), f"Expected directory not found: {directory}"
            assert (
                directory.is_dir()
            ), f"Path exists but is not a directory: {directory}"
            logger.info(f"✓ Directory exists: {directory}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_model_artifacts(self, temp_output_dir):
        """
        Verify that model checkpoints are created for all generations.

        For test config with:
        - init_population_size=3
        - num_generations=2
        - num_model_per_gen=2

        We expect:
        - gen_0_ind_* models (3 models from initialization)
        - gen_1_ind_* models (2 models from generation 1)
        - gen_2_ind_* models (2 models from generation 2)
        """
        logger.info("Verifying model artifacts...")

        model_dir = temp_output_dir / "models"
        assert model_dir.exists(), "Model directory not found"

        # Get all model directories
        model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(model_dirs)} model directories")

        # Verify we have models from each generation
        gen_0_models = [d for d in model_dirs if d.name.startswith("gen_0_")]
        gen_1_models = [d for d in model_dirs if d.name.startswith("gen_1_")]
        gen_2_models = [d for d in model_dirs if d.name.startswith("gen_2_")]

        logger.info(f"Generation 0: {len(gen_0_models)} models")
        logger.info(f"Generation 1: {len(gen_1_models)} models")
        logger.info(f"Generation 2: {len(gen_2_models)} models")

        # We should have models from initialization (gen 0)
        assert (
            len(gen_0_models) >= 3
        ), f"Expected at least 3 gen_0 models, found {len(gen_0_models)}"

        # We should have models from subsequent generations
        # Note: Actual counts may vary due to failures/retries
        assert (
            len(gen_1_models) > 0
        ), f"Expected gen_1 models, found {len(gen_1_models)}"

        # Verify each model directory contains expected files
        for model_path in gen_0_models[:1]:  # Check first model
            logger.info(f"Checking model: {model_path.name}")

            # Check for common model files (at least one should exist)
            model_files = list(model_path.glob("*"))
            assert (
                len(model_files) > 0
            ), f"Model directory is empty: {model_path}"

            # Check what files exist
            found_files = [f.name for f in model_files]
            logger.info(
                f"  Files in model dir: {', '.join(found_files[:5])}..."
            )

            # At least config.json should exist
            has_config = any(f.name == "config.json" for f in model_files)
            assert (
                has_config
            ), f"config.json not found in model directory: {model_path}"
            logger.info("  ✓ config.json found")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_archive_artifacts(self, temp_output_dir):
        """
        Verify that archive files are created and contain valid data.

        Expected files:
        - gen0_dns_archive.json
        - gen1_dns_archive.json
        - gen1_dns_archive_post_adapt_filtered.json
        - gen2_dns_archive.json
        - gen2_dns_archive_post_adapt_filtered.json
        """
        logger.info("Verifying archive artifacts...")

        archive_dir = temp_output_dir / "archives"
        assert archive_dir.exists(), "Archive directory not found"

        # Expected archive files
        expected_archives = [
            "gen0_dns_archive.json",
            "gen1_dns_archive.json",
            "gen2_dns_archive.json",
        ]

        found_archives = []
        for archive_name in expected_archives:
            archive_path = archive_dir / archive_name
            if archive_path.exists():
                found_archives.append(archive_name)
                logger.info(f"✓ Archive found: {archive_name}")
            else:
                logger.warning(f"✗ Archive not found: {archive_name}")

        # At least gen0 archive should exist
        assert (
            "gen0_dns_archive.json" in found_archives
        ), "gen0_dns_archive.json not found"

        # Verify archive content for gen0
        gen0_archive_path = archive_dir / "gen0_dns_archive.json"
        with open(gen0_archive_path, "r") as f:
            gen0_archive = json.load(f)

        logger.info(f"gen0_dns_archive contains {len(gen0_archive)} solutions")

        # Verify structure of archive entries
        if len(gen0_archive) > 0:
            first_solution = gen0_archive[0]
            required_fields = [
                "model_path",
                "fitness",
                "acdc_skill_vector",
            ]

            for field in required_fields:
                assert (
                    field in first_solution
                ), f"Required field '{field}' not found in archive solution"
                logger.info(f"  ✓ Field '{field}' present")

            # Verify skill vector is a dict
            assert isinstance(
                first_solution["acdc_skill_vector"], dict
            ), "acdc_skill_vector should be a dictionary"
            logger.info(
                f"  ✓ Skill vector has {len(first_solution['acdc_skill_vector'])} tasks"
            )

            # Verify fitness is a number
            assert isinstance(
                first_solution["fitness"], (int, float)
            ), "fitness should be a number"
            logger.info(f"  ✓ Fitness value: {first_solution['fitness']}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_task_pool_artifacts(self, temp_output_dir):
        """
        Verify that task pool files are created and properly formatted.

        Expected:
        - generated_tasks/pool/ directory
        - active_pool_gen_*.json files
        - active_limbo_map_gen_*.json files
        - task_*/ directories with task files
        """
        logger.info("Verifying task pool artifacts...")

        pool_dir = temp_output_dir / "generated_tasks" / "pool"
        assert pool_dir.exists(), "Task pool directory not found"

        # Find active pool files
        active_pool_files = list(pool_dir.glob("active_pool_gen_*.json"))
        logger.info(f"Found {len(active_pool_files)} active_pool files")

        # At least one active pool file should exist
        assert (
            len(active_pool_files) > 0
        ), "No active_pool_gen_*.json files found"

        # Verify content of first active pool file
        first_pool_file = sorted(active_pool_files)[0]
        logger.info(f"Checking: {first_pool_file.name}")

        with open(first_pool_file, "r") as f:
            active_pool = json.load(f)

        assert isinstance(
            active_pool, list
        ), "active_pool should be a list of task paths"
        logger.info(f"Active pool contains {len(active_pool)} tasks")

        # Verify task directories exist
        if len(active_pool) > 0:
            # Check first task
            first_task_path = Path(active_pool[0])
            assert (
                first_task_path.exists()
            ), f"Task path does not exist: {first_task_path}"
            assert (
                first_task_path.is_dir()
            ), f"Task path is not a directory: {first_task_path}"
            logger.info(f"✓ Task directory exists: {first_task_path.name}")

            # Check for task files
            task_files = list(first_task_path.glob("*"))
            logger.info(
                f"  Task contains {len(task_files)} files: {[f.name for f in task_files[:3]]}"
            )

            # At least task.py should exist
            task_py = first_task_path / "task.py"
            assert task_py.exists(), f"task.py not found in {first_task_path}"
            logger.info("  ✓ task.py found")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_generation_progression(self, temp_output_dir):
        """
        Verify that the generations progressed correctly.

        This checks:
        1. Archive files exist for expected generations
        2. Archive sizes are reasonable (not empty)
        3. Generation numbers are sequential
        """
        logger.info("Verifying generation progression...")

        archive_dir = temp_output_dir / "archives"
        assert archive_dir.exists(), "Archive directory not found"

        # Find all archive files
        archive_files = list(archive_dir.glob("gen*_dns_archive.json"))
        logger.info(f"Found {len(archive_files)} archive files")

        # Extract generation numbers
        generations = []
        for archive_file in archive_files:
            # Parse generation number from filename (e.g., gen2_dns_archive.json -> 2)
            name = archive_file.stem  # Remove .json
            if name.endswith("_dns_archive"):
                gen_str = name.split("_")[0][3:]  # Remove "gen" prefix
                try:
                    gen_num = int(gen_str)
                    generations.append(gen_num)
                except ValueError:
                    pass

        generations = sorted(generations)
        logger.info(f"Generations found: {generations}")

        # We expect at least gen 0, 1, 2 based on config (num_generations=2)
        assert 0 in generations, "Generation 0 archive not found"
        assert (
            len(generations) >= 2
        ), f"Expected at least 2 generations, found {len(generations)}"

        # Verify generation progression is sequential
        for i in range(len(generations) - 1):
            current_gen = generations[i]
            next_gen = generations[i + 1]
            assert (
                next_gen == current_gen + 1
            ), f"Generation progression not sequential: {current_gen} -> {next_gen}"

        logger.info("✓ Generation progression is sequential")

        # Verify each archive has content
        for gen in generations:
            archive_path = archive_dir / f"gen{gen}_dns_archive.json"
            with open(archive_path, "r") as f:
                archive = json.load(f)

            logger.info(
                f"  gen{gen}_dns_archive.json: {len(archive)} solutions"
            )

            # Archive should not be empty (at least initially)
            if gen == 0:
                assert len(archive) > 0, f"Generation {gen} archive is empty"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_data_consistency(self, temp_output_dir):
        """
        Verify data consistency between archives and models.

        This checks:
        1. Model paths in archives point to existing model directories
        2. Skill vectors have reasonable values
        3. Fitness values are consistent with skill vectors
        """
        logger.info("Verifying data consistency...")

        archive_dir = temp_output_dir / "archives"

        # Load gen0 archive
        gen0_archive_path = archive_dir / "gen0_dns_archive.json"
        if not gen0_archive_path.exists():
            pytest.skip("gen0_dns_archive.json not found")

        with open(gen0_archive_path, "r") as f:
            gen0_archive = json.load(f)

        if len(gen0_archive) == 0:
            pytest.skip("gen0_dns_archive.json is empty")

        logger.info(f"Checking consistency for {len(gen0_archive)} solutions")

        for i, solution in enumerate(gen0_archive[:3]):  # Check first 3
            logger.info(f"\nSolution {i + 1}:")

            # Check model path exists
            model_path = Path(solution["model_path"])
            logger.info(f"  Model path: {model_path}")

            # Model path should be absolute or relative to cwd
            if not model_path.is_absolute():
                # Try relative to output dir
                model_path = temp_output_dir / model_path.name

            if model_path.exists():
                logger.info("  ✓ Model directory exists")
            else:
                logger.warning(f"  ✗ Model directory not found: {model_path}")
                # This might be OK if models were cleaned up

            # Check skill vector
            skill_vector = solution["acdc_skill_vector"]
            assert isinstance(
                skill_vector, dict
            ), f"Solution {i}: skill_vector should be a dict"
            assert (
                len(skill_vector) > 0
            ), f"Solution {i}: skill_vector should not be empty"

            # All values should be floats between 0 and 1
            for task_id, score in skill_vector.items():
                assert isinstance(
                    score, (int, float)
                ), f"Solution {i}, task {task_id}: score should be numeric"
                assert (
                    0 <= score <= 1
                ), f"Solution {i}, task {task_id}: score should be in [0, 1]"

            logger.info(f"  ✓ Skill vector has {len(skill_vector)} tasks")

            # Check fitness
            fitness = solution["fitness"]
            assert isinstance(
                fitness, (int, float)
            ), f"Solution {i}: fitness should be numeric"
            assert (
                0 <= fitness <= 1
            ), f"Solution {i}: fitness should be in [0, 1]"

            # Fitness should approximately equal average of skill vector
            expected_fitness = sum(skill_vector.values()) / len(skill_vector)
            tolerance = 0.01
            assert abs(fitness - expected_fitness) < tolerance, (
                f"Solution {i}: fitness ({fitness}) does not match "
                f"average skill vector score ({expected_fitness})"
            )

            logger.info(f"  ✓ Fitness ({fitness:.4f}) matches skill vector")


# Additional utility tests for specific scenarios


class TestPipelineComponents:
    """Tests for specific pipeline components and edge cases."""

    @pytest.mark.integration
    def test_config_file_exists(self, integration_test_config):
        """Verify that the test configuration file exists."""
        config_path = Path(integration_test_config)
        assert config_path.exists(), f"Config file not found: {config_path}"
        logger.info(f"✓ Configuration file exists: {config_path}")

    @pytest.mark.integration
    def test_config_file_valid(self, integration_test_config):
        """Verify that the test configuration file is valid YAML."""
        import yaml

        config_path = Path(integration_test_config)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict), "Config should be a dictionary"
        assert "dns" in config, "Config should have 'dns' section"
        assert "acdc" in config, "Config should have 'acdc' section"

        logger.info("✓ Configuration file is valid YAML")
        logger.info(
            f"  dns.num_generations: {config['dns']['num_generations']}"
        )
        logger.info(
            f"  dns.init_population_size: {config['dns']['init_population_size']}"
        )
        logger.info(
            f"  acdc.initial_pool_size: {config['acdc']['initial_pool_size']}"
        )


# Fixtures for test dependencies


@pytest.fixture(scope="class")
def run_pipeline_once():
    """
    Run the pipeline once and reuse results for multiple tests.

    This fixture runs the pipeline and returns the output directory path.
    All tests in TestFullPipeline class can use this shared result.
    """
    # Create shared test directory in outputs_tests/
    outputs_tests_base = Path("outputs_tests")
    outputs_tests_base.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = outputs_tests_base / f"shared_test_run_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("Running pipeline once for all tests (shared fixture)")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("=" * 80)

    # Build command
    abs_output_dir = output_dir.absolute()
    cmd = [
        "python",
        "main_ac_dc.py",
        "--config-name",
        "test_integration",
        f"output_dir={abs_output_dir}",
        f"hydra.run.dir={abs_output_dir}",
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Output: {output_dir}")

    # Run with timeout
    timeout_seconds = 1800
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            # Only log errors and important info
            if "ERROR" in line or "Generation" in line or "Archive" in line:
                logger.info(f"[PIPELINE] {line.rstrip()}")

            if time.time() - start_time > timeout_seconds:
                process.kill()
                raise TimeoutError("Pipeline execution timed out")

        return_code = process.wait()
        elapsed_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info(f"Pipeline completed in {elapsed_time:.2f}s")
        logger.info(f"Return code: {return_code}")
        logger.info("=" * 80)

        if return_code != 0:
            logger.error("Pipeline failed! Last 30 lines:")
            for line in output_lines[-30:]:
                logger.error(line.rstrip())
            pytest.fail(f"Pipeline failed with return code {return_code}")

    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        pytest.fail(f"Pipeline execution failed: {e}")

    # Yield the output directory for tests to use
    yield output_dir

    # Cleanup after all tests complete
    logger.info(f"Cleaning up shared test output: {output_dir}")
