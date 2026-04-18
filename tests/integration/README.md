# Integration Tests

Tests for the AC/DC pipeline covering multiple components working together.

## Test Files

### `test_full_pipeline.py`

Runs the complete AC/DC pipeline with a lightweight config and verifies all expected artifacts.

**TestFullPipeline:**
- `test_full_pipeline_execution` — complete pipeline run
- `test_directory_structure` — all output directories created
- `test_model_artifacts` — model checkpoints exist
- `test_archive_artifacts` — archive JSON files created and valid
- `test_task_pool_artifacts` — task pool files present
- `test_generation_progression` — correct generation sequence
- `test_data_consistency` — consistency between component outputs

**TestPipelineComponents:**
- `test_config_file_exists` / `test_config_file_valid` — validates `configs/test_integration.yaml`

**Config (`configs/test_integration.yaml`):** 3 seed models, 2 generations, 2 models/gen, pool size 5, solo mode, W&B disabled.

**Requirements:** GPU (`@pytest.mark.requires_gpu`), Docker for sandbox, solo mode or Redis/RabbitMQ for Celery.

**Runtime:** ~15-30 min for full pipeline; individual verification tests <1s.

### `test_global_skill_vector.py`

Tests the end-to-end skill vector evaluation pipeline.

- Skill vector file I/O and directory structure
- Fitness calculation (average of skill vector)
- Model filtering by generation and seed model detection
- Coverage-based and fitness-based model selection

## Running

```bash
# Full pipeline (slow, requires GPU)
pytest tests/integration/test_full_pipeline.py -v -s

# Fast integration tests only
pytest tests/integration/ -m "not slow and not requires_gpu" -v

# Skill vector tests
pytest tests/integration/test_global_skill_vector.py -v

# With debug logging
pytest tests/integration/test_full_pipeline.py -v -s --log-cli-level=DEBUG
```

## Expected Artifacts

After `test_full_pipeline.py`, output appears in `outputs_tests/test_run_<timestamp>/` (covered by `.gitignore`):

```
outputs_tests/
└── test_run_<timestamp>/
    ├── models/
    │   ├── gen_0_ind_0/ ... gen_0_ind_2/
    │   ├── gen_1_ind_0/ ... gen_1_ind_1/
    │   └── gen_2_ind_0/ ... gen_2_ind_1/
    ├── archives/
    │   ├── gen0_dns_archive.json
    │   ├── gen1_dns_archive.json
    │   ├── gen1_dns_archive_post_adapt_filtered.json
    │   ├── gen2_dns_archive.json
    │   └── gen2_dns_archive_post_adapt_filtered.json
    └── generated_tasks/pool/
        ├── active_pool_gen_{0,1,2}.json
        ├── active_limbo_map_gen_{0,1,2}.json
        └── task_*/
```

## Troubleshooting

| Issue | Fix |
|---|---|
| Test timeout | Reduce generations/models in config, check GPU availability |
| Model loading failure | Verify seed model paths in config, check GPU memory |
| Celery connection error | Start Redis (`redis-server`) or use solo mode |
| Docker errors | Verify Docker is running, check memory limits |
| Missing archive files | Check generation logs for errors, verify disk space |
