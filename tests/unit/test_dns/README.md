# DNS Unit Tests

Unit tests for the Dominated Novelty Search (DNS) components: archive management, novelty metrics, skill vectors, and coverage metrics.

## Test Files

### `test_dns_utils.py`
Tests for DNS archive management and skill vector utilities.

- **TestSkillVectorCreation** (5 tests): binary and QD-mode skill vectors, DNS solution creation
- **TestACDCSkillVectors** (4 tests): AC/DC solutions, format conversions
- **TestArchivePersistence** (3 tests): DNS archive save/load with round-trip verification
- **TestACDCArchivePersistence** (5 tests): AC/DC archive with eval details and truncation
- **TestHammingDistance** (5 tests): Hamming distance with length mismatch validation

### `test_novelty_metrics.py`
Tests for dominated novelty score computation.

- **TestDifficultyWeights** (5 tests): difficulty weights from population failure rates
- **TestDominatedNoveltyBasic** (3 tests): novelty scoring with Hamming distance
- **TestDominatedNoveltyWithSkillRatio** (4 tests): skill ratio = unique_solved / total_unsolved_by_stronger
- **TestDominatedNoveltyWithDifficultyWeights** (2 tests): weighted novelty scores
- **TestDominatedNoveltyKNeighbors** (3 tests): K-nearest neighbors limiting for scalability
- **TestDominatedNoveltySubsetSkillVector** (2 tests): skill vector subsetting for efficiency

### `test_archive_update.py`
Tests for DNS archive update strategies.

- **TestTopFitnessSelection** (6 tests): fitness-based selection, weak solution rejection
- **TestDNSArchiveUpdate** (8 tests): novelty-based selection with various configurations
- **TestArchiveUpdateEdgeCases** (6 tests): single solutions, identical fitness, perfect/zero fitness
- **TestArchiveUpdateDeterminism** (2 tests): identical results across repeated runs

### `test_metrics.py`
Tests for coverage and quality metrics (`dns/metrics.py`).

- **TestComputeACDCoverageMetrics**: per-generation coverage metrics — `compute_acdc_coverage_metrics` with varying model counts, thresholds, and score distributions
- **TestAnalyzeCombinedCoverage**: archive-level aggregation — `analyze_combined_coverage` verifying rollup keys and edge cases (empty archive, all-pass, all-fail)

## Fixtures (`conftest.py`)

**VectorDB:** `mock_embedding_client`, `vector_db_with_mock_embedding`, `vector_db_with_samples`, `vector_db_with_diverse_samples`, `vector_db_with_many_samples`

**DNS solutions:** `dns_solutions`, `dns_population`, `ac_dc_solutions`, `dns_archive`, `new_dns_solutions`

**DNS config:** `dns_cfg_basic`, `dns_cfg_top_fitness`, `dns_cfg_novelty`

**Archive variants:** `high_fitness_archive`, `low_fitness_solutions`, `many_dns_solutions`, `dns_archive_diverse`, `new_dns_solutions_novel`

**Mock data:** `mock_task_metrics`, `mock_task_metrics_qd`, `mock_tasks`, `mock_acdc_archive_data_for_metrics`, `mock_tasks_for_metrics`

## Running the Tests

```bash
# All DNS tests
pytest tests/unit/test_dns/ -v

# Individual files
pytest tests/unit/test_dns/test_dns_utils.py -v
pytest tests/unit/test_dns/test_novelty_metrics.py -v
pytest tests/unit/test_dns/test_archive_update.py -v
pytest tests/unit/test_dns/test_metrics.py -v

# With coverage
pytest tests/unit/test_dns/ --cov=dns --cov-report=html
```
