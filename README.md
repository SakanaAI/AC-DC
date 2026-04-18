# AC/DC

Assessment Coevolving with Diverse Capabilities (AC/DC) is a system framework for coevolving populations of large language models (LLMs) and synthetic tasks that challenge them. The system coordinates task generation, model crossover/mutation, distributed evaluation, and archive management to surface models that excel on diverse, emerging capabilities.

## Repository Layout

- `main_ac_dc.py` – Hydra-driven entry point that orchestrates the AC/DC optimization loop, Celery workers, and W&B logging.
- `configs/` – Hydra configuration tree for Dominated Novelty Search (DNS) model archive management, task generation, Celery infra, fractional GPU scheduling, and evolution operators.
- `tasks/` – Task generation toolkit, prompts, and sandbox integrations.
- `seed_tasks/` – Seed tasks used to bootstrap the active task pool.
- `dns/` – DNS utilities including archive management, and metrics.
- `crossover/` & `mutation/` – Evolution operators for combining and perturbing model checkpoints.
- `workers/` – Celery worker implementations for model initialization, merging, and evaluation.
- `evaluation/` – Offline analysis notebooks and scripts for benchmark-style assessments of evolved models.
- `utils/` – Shared helpers for Celery setup, filesystem management, sandbox configuration, and infrastructure glue.
- `frac_start_workers.sh` – Convenience script that launches a tmux session with fractional GPU workers plus the coordinating process.

## Install Dependencies

```bash
conda create -n acdc python=3.11 -y
conda activate acdc
pip install --upgrade pip
pip install -r requirements.txt
```

## Running an Experiment

1. **Configure credentials & services**
   - Provide access to model checkpoints referenced in `configs/ac_dc.yaml` (`seed_model_paths`).
   - Set the W&B entity (`wandb.entity`) in `ac_dc.yaml` to your W&B entity.
   - In `frac_start_workers.sh`, set the `OUTPUT_DIR` to the path where you want to store the experiment results. If `""`, the output sdir will be set in the current working directory.
   - Ensure the docker sandbox is built by running `docker build -t acdc-sandbox:latest docker/sandbox/`
   - Ensure Celery broker/backends (RabbitMQ/Redis by default) are reachable or override via Hydra CLI arguments by running `bash docker_setup.sh`

2. **Ensure the relevant served models are available**
    - Embedding model: `CUDA_VISIBLE_DEVICES=0; python -m vllm.entrypoints.openai.api_server --model intfloat/e5-mistral-7b-instruct --served-model-name e5-mistral-7b-instruct --task embedding --port 8010`
    - Scientist model: `CUDA_VISIBLE_DEVICES=0,1,2,3; python -m vllm.entrypoints.openai.api_server --served-model-name Qwen/Qwen2.5-72B-Instruct --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 --port 8001 --enable-prefix-caching --gpu-memory-utilization 0.8`

3. **Set the IP addresses for the vLLM servers**
   - To get the address for `vllm_server_host` in `acdc/default.yaml` and `vllm_embedding_server_host` in `task_generation/default.yaml`, run `ip a` on the node with the running big model vLLM instance, and get the IP address from second group of rows with "enp0s12" in "inet".
   - Set the IP addresses in `acdc/default.yaml` (`vllm_server_host`) and `task_generation/default.yaml` (`vllm_embedding_server_host`).

4. **Launch distributed workers and training** \
   Set `NUM_WORKERS` in `frac_start_workers.sh` depending on the number of requested GPUs (one worker per GPU for 7B models)
   Use `frac_start_workers.sh` to spawn Celery workers with fractional GPU assignments

   ```bash
   ./frac_start_workers.sh --gpus=2 --workers-per-gpu=2 --session=ac_dc
   ```

## Evaluating the Experiment

0. **Install the AC-DC-eval_harness and dependencies** \
   Eval harness (minerva math task) has a dependency that is incompatible with the acdc conda environment. So, we create a new conda environment for the eval harness and add the dependencies there.

   ```bash
   conda create -n acdc_eval python=3.11 -y
   conda activate acdc_eval
   pip install -r requirements.txt

   git clone https://github.com/SakanaAI/AC-DC-eval_harness.git

   cd AC-DC-eval_harness
   pip install -e .
   pip install -e ".[math]"
   ```

1. **Perform global task pool evaluation** \
   Run the following script to evaluate the global task pool against the archived models:
   ```bash
   python global_task_pool_eval.py output_dir=/path/to/output_dir base_model_path=Qwen/Qwen2-7B-Instruct acdc.num_sandbox_workers=64 vllm_pop.max_tokens=1024
   ```

2. **Run AC-DC-eval_harness evaluation**
   > NOTE: This step requires the AC-DC-eval_harness to be installed and your `acdc_eval` conda environment to be activated. See step 0 above.

   > NOTE 2: If you don't want to evaluate all models in the archive, you can set the MODEL_PATHS in `evaluate_archive_lm_harness.sh` bash script. (See @Misc section below)
   
   a. Benchmarks w/o code evaluation:
      - Configure BASE_MODEL and LLM_AS_A_JUDGE_MODEL_URL in `evaluate_archive_lm_harness.sh` bash script
      ```bash
      bash evaluation/evaluate_archive_lm_harness.sh --output_dir /path/to/output_dir
      ```
   b. Benchmarks w/ code evaluation:
      - Configure BASE_MODEL in `evaluate_archive_lm_harness_docker.sh` bash script
      ```bash
      bash evaluation/evaluate_archive_lm_harness_docker.sh --output_dir /path/to/output_dir
      ```

3. **Compute the Coverage Metrics** \
   Run the following scripts (here, with example configuration for a Qwen2-7B run) to compute the coverage metrics: \
   a. Non-code benchmarks:
   ```bash
   python evaluation/coverage.py -e path/to/output_dir --selection_methods_config selection_methods_main-qwen2-7B.yaml
   ```
   b. Code benchmarks:
   ```bash
   python evaluation/coverage.py -e path/to/output_dir --selection_methods_config selection_methods_main-qwen2-7B.yaml -l lm_harness_code -bm benchmarks_code.yaml 
   ```

4. **Compute Best-of-N scores** \
   Run the following scripts to compute the best-of-N scores: \
   a. Non-code benchmarks using LLM-as-a-judge based methods:
   ```bash
   LLM_AS_A_JUDGE_MODEL_URL=http://xxx.xxx.xxx.xxx:8001/v1 python evaluation/single_answer_from_pop_analysis.py -e path/to/output_dir --num_workers 64
   ```
   b. Code benchmarks using Reward Model based methods: \
   (For N=8 models)
   
   ```bash
   python evaluation/single_answer_from_pop_rm_based.py -e path/to/output_dir -n 8 -d "cuda:0"
   ```
   (For N=3 models)

   ```bash
   python evaluation/single_answer_from_pop_rm_based.py -e path/to/output_dir -n 3 -d "cuda:1"
   ```

## Baselines Evaluation

Run the following scripts in order to evaluate baselines. The example below uses the `qwen2.5` model family (8× `Qwen/Qwen2.5-7B-Instruct`) with start seeds `50` and `58`.

1. **Run LM Evaluation Harness** — non-code benchmarks (using the `acdc_eval` conda environment): \
   In `run_baselines-eval_harness.sh`, set:
   - `LLM_AS_A_JUDGE_MODEL_URL` — URL of your running judge model server \

   In `evaluation/evaluate_baselines_lm_harness_control.sh`, set:
   - `LLM_AS_A_JUDGE_MODEL_NAME` — judge model name (default: `Qwen/Qwen2.5-72B-Instruct`)
   - `GPUS` — GPUs to use (default: `0,1,2,3`); models are evaluated in parallel across GPUs
   - `MODELS` — list of baseline models to evaluate (repeat entries for multiple seeds)
   ```bash
   bash run_baselines-eval_harness.sh
   ```
   Results are saved to `outputs/baselines/qwen2.5/{seed}/lm_harness/`.

2. **Run LM Evaluation Harness via Docker** — code benchmarks (`humaneval`, `mbpp`) with sandboxed execution (run with a different conda env so that `CONDA_ENV_NAME` is recognized): \
   In `run_baselines-eval_harness-docker.sh`, set:
   - `DOCKER_GPUS` — GPUs to use (default: `0,1,2,3`) \

   In `evaluation/evaluate_baselines_lm_harness_docker_control.sh`, set:
   - `LM_EVAL_REPO_PATH` — path to the local `AC-DC-eval_harness` repo (default: `./AC-DC-eval_harness`); the Docker image (`lm-eval-sandbox`) is auto-built from its `Dockerfile` if not already present
   - `MODELS` — list of baseline models to evaluate
   - `DOCKER_MEMORY` / `DOCKER_CPUS` — container resource limits (default: `128g` / `128`)
   ```bash
   bash run_baselines-eval_harness-docker.sh
   ```
   Results are saved to `outputs/baselines/qwen2.5/{seed}/lm_harness_code/`.

3. **Clean up output directories** — flattens result files from the nested `<model>/seed_XX/<model>/` layout produced by lm-harness into `<model>/seed_XX/`: \
   Use `--dry-run` to preview changes before applying them.
   ```bash
   bash run_baselines-dir_cleanup.sh
   ```

4. **Compute coverage metrics** — uses `qwen2_5-7B-control.yaml` as the model names config, for both `lm_harness` (non-code) and `lm_harness_code` (code) benchmarks:
   ```bash
   bash run_baselines-metrics-coverage.sh
   ```

5. **Compute Best-of-N metrics** — uses `qwen2_5-7B-control.yaml`; set `LLM_AS_A_JUDGE_MODEL_URL` in the script for LLM-as-a-judge scoring on non-code benchmarks (N=8 and N=3), and runs RM-based scoring for code benchmarks (N=8 and N=3):
   ```bash
   bash run_baselines-metrics-bon.sh
   ```

## Visualization
We have developed two streamlit-based visualization tools to help you visualize the evolution of the model archive and the task pool.
- `visualization/show_evolution_of_tasks.py` - This tool allows you to visualize the evolution of the task pool over generations.
- `visualization/show_evolution_of_models.py` - This tool allows you to visualize the evolution of the model archive over generations.

To run the visualization tools, run the following commands:
```bash
streamlit run visualization/show_evolution_of_tasks.py <path_to_experiment_dir>
streamlit run visualization/show_evolution_of_models.py <path_to_experiment_dir>,<path_to_baselines_results_dir>
```

> Note: The baselines results directory is optional. If not provided, the tool (should) simply not plot any baseline results.

> Also note: Both scripts are not tested for all edge cases. Please report any issues you encounter.

## Misc

- How to determine the relevant models for evaluation:
  - Run the following script:
  ```bash
  python evaluation/coverage.py -e path/to/output_dir --selection_methods_config selection_methods_main-qwen2-7B.yaml --list_models_only
  ```
  This will give you the list of models that are considered for evaluation. You can then set the MODEL_PATHS in `evaluate_archive_lm_harness.sh` bash script to the list of models.
  
## Acknowledgments

We thank the authors that released [Dominated Novelty Search (DNS)](https://arxiv.org/abs/2502.00593) and [Automated Capability Discovery (ACD)](https://arxiv.org/abs/2502.07577) for inspiring research and development of our AC/DC framework.