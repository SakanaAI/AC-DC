# Benchmark Question Vector Database

A system for creating and querying vector databases of questions from major AI benchmarks and synthetically generated tasks.

## Overview

This system provides two main functionalities:

1. **Benchmark Question Database**: Loads questions from various AI benchmarks, formats them appropriately, and stores them in a semantic vector database for similarity search and retrieval.

2. **Synthetic Task Database**: Loads synthetically generated tasks from experiment outputs and creates a vector database based on their example instructions.

### Supported Benchmarks

1. **MMLU** - Massive Multitask Language Understanding (57 subjects, ~570 samples)
2. **MMLU Pro** - Advanced MMLU (14 categories, ~140 samples)
3. **Big Bench Hard** - Reasoning tasks (27 tasks, ~270 samples)
4. **MATH** - Competition math problems (7 categories, ~70 samples)
5. **GSM8K** - Grade school math (30 samples)
6. **HumanEval** - Code generation (30 samples)
7. **MBPP** - Python programming (30 samples)
8. **GPQA** - Graduate-level science questions (3 domains, ~30 samples)

**Total: ~1,170 question samples**

## Installation

### Prerequisites

```bash
# Install required packages
pip install datasets huggingface_hub numpy scikit-learn openai
```

### Directory Structure

```
benchmark_question_db/
├── simple_vectordb.py                      # Vector database implementation
├── loaders/                                # Benchmark data loaders
│   ├── __init__.py
│   ├── base.py                             # Base classes
│   ├── mmlu.py
│   ├── mmlu_pro.py
│   ├── bbh.py
│   ├── math_dataset.py
│   ├── gsm8k.py
│   ├── humaneval.py
│   └── mbpp.py
├── utils/
│   └── db_explorer.py                      # Database exploration utilities
├── build_benchmark_vectordb.py             # Benchmark database build script
├── build_synthetic_vectordb.py             # Synthetic task database build script
├── benchmark_db_config.yaml                # Configuration file
├── README.md                               # This file
├── PLANNING.md                             # Planning document
├── SYNTHETIC_TASK_VECTORDB_PLAN.md         # Synthetic task planning document
└── DATASET_STRUCTURES.md                   # Dataset structure reference
```

## Usage

### 1. Building the Benchmark Question Database

#### With Mock Embeddings (for testing/development)

```bash
# Use mock embeddings (no embedding server required)
python benchmark_question_db/build_benchmark_vectordb.py \
    --storage-path ./benchmark_vector_db \
    --mock-embeddings \
    --verbose
```

#### With Real Embeddings

First, start an embedding server:

```bash
# Example with vLLM (in a separate terminal)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --served-model-name all-MiniLM-L6-v2 \
    --task embedding \
    --port 8010
```

Then build the database:

```bash
python benchmark_question_db/build_benchmark_vectordb.py \
    --storage-path ./benchmark_vector_db \
    --embedding-model all-MiniLM-L6-v2 \
    --embedding-url http://localhost:8010/v1 \
    --verbose
```

#### Command-line Options

```
--storage-path         Path to store the database (default: ./benchmark_vector_db)
--embedding-model      Name of embedding model (default: all-MiniLM-L6-v2)
--embedding-url        URL of embedding server (default: http://localhost:8010/v1)
--n-per-subgroup       Samples per subgroup (default: 10)
--total-samples        Samples for benchmarks without subgroups (default: 30)
--random-seed          Random seed for reproducibility (default: 42)
--include-mcq-options  Include multiple-choice options in question formatting (default: False)
--mock-embeddings      Use mock embeddings for development
--verbose, -v          Enable verbose logging
```

**Note on Multiple-Choice Question Formatting:**

By default, multiple-choice questions (from MMLU, MMLU Pro) are embedded **without** the answer choices - only the question text is used. This provides better semantic search results when queries don't include answer choices.

To include answer choices in embeddings (legacy behavior), use the `--include-mcq-options` flag:

```bash
python benchmark_question_db/build_benchmark_vectordb.py \
    --storage-path ./benchmark_vector_db \
    --include-mcq-options \
    --mock-embeddings
```

**Invalid Sample Filtering:**

The system automatically filters out invalid samples before creating the database. Invalid samples are stored in `benchmark_question_db/invalid_samples_per_benchmark/` and include questions that:
- Are incomplete or poorly formatted
- Cannot be answered without multiple-choice options
- Have ambiguous or incorrect answers

Filtering is applied to: MMLU, MMLU Pro, and Big Bench Hard. The system logs how many invalid samples were filtered from each subgroup/category.

### 2. Building the Synthetic Task Database

The synthetic task database is built from experiment outputs containing synthetically generated tasks.

#### With Mock Embeddings (for testing/development)

```bash
# Use mock embeddings (no embedding server required)
python -m benchmark_question_db.build_synthetic_vectordb \
    --experiment-dir outputs/qwen2.5 \
    --storage-path qwen2.5_synth_task-vectordb \
    --mock-embeddings \
    --verbose
```

#### With Real Embeddings

First, start an embedding server (same as for benchmark database):

```bash
# Example with vLLM (in a separate terminal)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --served-model-name all-MiniLM-L6-v2 \
    --task embedding \
    --port 8010
```

Then build the database:

```bash
python -m benchmark_question_db.build_synthetic_vectordb \
    --experiment-dir outputs/qwen2.5 \
    --storage-path qwen2.5_synth_task-vectordb \
    --embedding-model all-MiniLM-L6-v2 \
    --embedding-url http://localhost:8010/v1 \
    --verbose
```

#### Synthetic Task Command-line Options

```
--experiment-dir       Path to experiment directory (default: outputs/qwen2.5)
--storage-path         Path to store the database (default: qwen2.5_synth_task-vectordb)
--embedding-model      Name of embedding model (default: all-MiniLM-L6-v2)
--embedding-url        URL of embedding server (default: http://localhost:8010/v1)
--mock-embeddings      Use mock embeddings for development
--verbose, -v          Enable verbose logging
```

#### Synthetic Task Directory Structure

The script expects the following structure:

```
{experiment_dir}/
└── generated_tasks/
    └── pool/
        ├── task_1_probability_card_draw/
        │   └── task.json
        ├── task_2_logical_contrapositive/
        │   └── task.json
        └── ...
```

Each `task.json` file should contain:

```json
{
    "name_of_task": "logic_translation",
    "description_of_task": "Translate natural language statements...",
    "capability_being_measured": "logical_reasoning",
    "estimated_human_difficulty": "3",
    "example_instruction": "Translate the following natural language statement..."
}
```

The `example_instruction` field is used for creating embeddings.

### 3. Exploring the Database

The `db_explorer.py` utility provides several commands:

#### Show Statistics

```bash
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db stats
```

Output:
```
DATABASE STATISTICS
================================================================================

Total samples: 1140

By Benchmark:
  BigBenchHard: 270
  GSM8K: 30
  HumanEval: 30
  MATH: 70
  MBPP: 30
  MMLU: 570
  MMLU_Pro: 140

By Question Type:
  code_generation: 60
  math_problem: 70
  math_word_problem: 30
  multiple_choice: 710
  reasoning: 270
```

#### List Samples

```bash
# List all samples (first 10)
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db list

# List samples from specific benchmark
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db list \
    --benchmark MMLU --limit 20

# Show content
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db list \
    --show-content --limit 5
```

#### Search for Similar Questions

```bash
# Search for similar questions
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db search \
    "What is the capital of France?"

# Search within specific benchmark
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db search \
    "How to reverse a string in Python?" \
    --benchmark HumanEval

# Get more results
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db search \
    "algebra problem about quadratic equations" \
    --top-n 10
```

#### Get Specific Sample

```bash
python benchmark_question_db/utils/db_explorer.py --db-path ./benchmark_vector_db get \
    mmlu_abstract_algebra_5
```

### 4. Using in Python Code

#### Benchmark Question Database

```python
from benchmark_question_db.simple_vectordb import SimpleVectorDB

# Load the database
db = SimpleVectorDB(
    storage_path="./benchmark_vector_db",
    task_representation_vector_db="content",
)

# Get statistics
print(f"Total samples: {db.get_count()}")

# Search for similar questions
results = db.find_similar(
    query="What is photosynthesis?",
    top_n=5,
    similarity_threshold=0.3,
)

for result in results:
    print(f"ID: {result['sample_id']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Benchmark: {result['metadata']['benchmark']}")
    print(f"Content: {result['content'][:100]}...")
    print()

# Get specific sample
sample = db.get_sample("mmlu_biology_10")
if sample:
    print(f"Question: {sample['content']}")
    print(f"Metadata: {sample['metadata']}")
```

#### Synthetic Task Database

```python
from benchmark_question_db.simple_vectordb import SimpleVectorDB

# Load the synthetic task database
db = SimpleVectorDB(
    storage_path="qwen2.5_synth_task-vectordb",
    task_representation_vector_db="content",
)

# Get statistics
print(f"Total tasks: {db.get_count()}")

# Search for similar tasks
results = db.find_similar(
    query="Solve a probability problem with cards",
    top_n=5,
    similarity_threshold=0.3,
)

for result in results:
    print(f"Task ID: {result['sample_id']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Task Name: {result['metadata']['name_of_task']}")
    print(f"Capability: {result['metadata']['capability_being_measured']}")
    print(f"Difficulty: {result['metadata']['estimated_human_difficulty']}")
    print(f"Instruction: {result['content'][:100]}...")
    print()

# Get specific task
task = db.get_sample("task_345_logic_translation")
if task:
    print(f"Instruction: {task['content']}")
    print(f"Metadata: {task['metadata']}")
```

## Question Formats

### Multiple Choice (MMLU, MMLU Pro)

```
Question: What is the capital of France?
A) London
B) Paris
C) Berlin
D) Madrid
```

### Math Problems (GSM8K, MATH)

```
Problem: If John has 5 apples and gives 2 to Mary, how many does he have left?
```

### Code Generation (HumanEval, MBPP)

```
Code Generation Problem:

def reverse_string(s: str) -> str:
    """ Reverse the input string.
    >>> reverse_string("hello")
    "olleh"
    """
```

## Metadata Structure

### Benchmark Question Metadata

Each benchmark sample includes metadata:

```json
{
    "benchmark": "MMLU",
    "subgroup": "abstract_algebra",
    "question_type": "multiple_choice",
    "answer_index": 1,
    "answer_letter": "B",
    "answer_text": "Paris",
    "num_choices": 4,
    "original_index": 42
}
```

### Synthetic Task Metadata

Each synthetic task includes metadata:

```json
{
    "task_id": "task_345_logic_translation",
    "name_of_task": "logic_translation",
    "description_of_task": "Translate natural language statements into formal logic expressions and vice versa.",
    "capability_being_measured": "logical_reasoning",
    "estimated_human_difficulty": "3"
}
```

## Configuration

Edit `benchmark_db_config.yaml` to customize:

```yaml
sampling:
  samples_per_subgroup: 10
  samples_no_subgroup: 30
  random_seed: 42

development:
  mock_embeddings: false
  verbose_logging: false
```

## Development and Testing

### Mock Embeddings Mode

For development and testing without an embedding server:

```bash
python benchmark_question_db/build_benchmark_vectordb.py --mock-embeddings
```

This generates reproducible random embeddings based on text content hash.

### Custom Loaders

To add a new benchmark:

1. Create a new loader in `benchmark_question_db/loaders/`
2. Inherit from `BenchmarkLoader`
3. Implement required methods:
   - `load_dataset()`
   - `get_subgroups()`
   - `format_question()`
   - `_sample_from_subgroup()` or `_sample_from_dataset()`

Example:

```python
from benchmark_question_db.loaders.base import BenchmarkLoader

class MyBenchmarkLoader(BenchmarkLoader):
    def load_dataset(self):
        return load_dataset("my/benchmark", split="test")

    def get_subgroups(self):
        return ["category1", "category2"]  # or None

    def format_question(self, item):
        return f"Question: {item['question']}"

    # ... implement other methods
```

## Troubleshooting

### Issue: Dataset Loading Fails

**Solution**: Some datasets may be gated or require authentication. Make sure you're logged in to HuggingFace:

```bash
huggingface-cli login
```

### Issue: Embedding Server Connection Error

**Solution**:
1. Check that the embedding server is running
2. Verify the URL is correct
3. Or use `--mock-embeddings` for testing

### Issue: Out of Memory

**Solution**: Process benchmarks separately or reduce `--n-per-subgroup`

## Dataset Sources

- **MMLU**: `cais/mmlu`
- **MMLU Pro**: `TIGER-Lab/MMLU-Pro`
- **Big Bench Hard**: `SaylorTwift/bbh`
- **MATH**: `EleutherAI/hendrycks_math`
- **GSM8K**: `openai/gsm8k`
- **HumanEval**: `openai/openai_humaneval`
- **MBPP**: `google-research-datasets/mbpp`
- **GPQA**: `Idavidrein/gpqa`

## Notes

- Questions only (not answers) are embedded for semantic search
- Answers and solutions are stored in metadata
- Random seed ensures reproducible sampling
- Database can be exported/imported as zip files

## License

This tool is part of the AC/DC project.
