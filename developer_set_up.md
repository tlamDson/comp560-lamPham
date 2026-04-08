# Developer Setup Guide

This document is the English onboarding reference for developers and AI agents working in this repository.

It complements project-specific READMEs by focusing on:
- Repository structure
- Canonical environment setup
- Standard run commands
- Reproducible dependency management
- Benchmark and troubleshooting practices

## 1. Project Overview

This repository combines arithmetic model experiments, shared training code, a NanoGPT fork, and local Phi model assets.

Main workspace directories:
- `comp560-lamPham/`: Task code and experiment folders (4 to 100 digit addition).
- `comp560-nanoGPT/`: NanoGPT training/model code used by the project.
- `Phi-3-mini-4k-instruct/`: Local Phi model files for Phi-based workflows.
- `scripts/`: Environment lock/install helper scripts.
- `docs/`: Additional documentation.

Environment and runtime files at repo root:
- `run.sh`: Canonical Python launcher wrapper.
- `validate_env.py`: Environment gatekeeper checks.
- `.env.example`: Local environment template.
- `requirements-core.txt`, `requirements-phi.txt`, `requirements-lock.txt`: Python dependency layers and lock.
- `environment.yml`, `conda-lock.yml`: Conda base spec and lock file.

## 2. Canonical Runtime Policy

Use these rules to avoid interpreter drift and machine-specific breakage:

1. Do not run `python ...` directly for normal project workflows.
2. Always run Python commands through `run.sh`.
3. Run `validate_env.py` before training/benchmark sessions.
4. Keep `.env` local and uncommitted.

Interpreter resolution in `run.sh` (priority order):
1. `ML_PYTHON_PATH` from `.env`
2. `ML_CONDA_ENV` activation path (if configured)
3. `.vscode/settings.json` interpreter path
4. Active conda environment (`CONDA_PREFIX`)
5. Local `.venv`
6. `python3` or `python` in shell `PATH`

## 3. First-Time Setup

From repository root:

```bash
cd /home/t1amp/comp560-LamPham
cp .env.example .env
```

Edit `.env` and set your canonical interpreter:

```env
ML_PYTHON_PATH=/home/your-user/miniconda3/envs/ai/bin/python
PHI_MODEL_PATH=Phi-3-mini-4k-instruct
```

Then run:

```bash
chmod +x run.sh
./run.sh --print-python
./run.sh validate_env.py --profile core --require-cuda
```

Optional profile checks:

```bash
./run.sh validate_env.py --profile bot --require-cuda
./run.sh validate_env.py --profile phi --require-cuda
./run.sh validate_env.py --profile phi-train --require-cuda
```

## 4. Dependency Management (Reproducible)

Install from existing lock files:

```bash
cd /home/t1amp/comp560-LamPham
chmod +x scripts/install_from_lock.sh
./run.sh -m pip install conda-lock
./scripts/install_from_lock.sh ai
```

Regenerate lock files when dependencies change:

```bash
cd /home/t1amp/comp560-LamPham
chmod +x scripts/lock_deps.sh
./run.sh -m pip install pip-tools conda-lock
./scripts/lock_deps.sh
```

## 5. Repository Structure Details

Important directories in `comp560-lamPham/`:
- `common/train.py`: Shared training entrypoint.
- `models/`: Model construction and wrappers (`nanogpt_model.py`, `phi_wrapper.py`, `model_factory.py`).
- `4_digits_addition/`: Primary tuning and benchmark reference folder.
- `5_digits_addition/` ... `100_digits_addition/`: Digit-specific experiment folders.

Each digit folder usually includes:
- `config/`: Training configurations.
- `data/`: Prepared dataset files.
- `sample_and_verify_linux.py`: Verification script.
- `bench_run.sh`: Automated 5-run benchmark.
- `README.md`: Folder-specific notes.

## 6. Data and Verification Standard (All Digit Folders)

All digit-addition folders must follow the same dataset and verification format to keep training and evaluation comparable.

Data generation policy (`data/basic/prepare.py`):
- Scenario mix is fixed at `70/10/10/10`:
  - `stratified_random`: 70%
  - `cascading_carries`: 10%
  - `extreme_imbalance`: 10%
  - `boundary_zeros`: 10%
- Examples use fixed-width operands and reversed targets:
  - Input format: `a+b=`
  - Target format: reversed, zero-padded sum
- `meta.pkl` must include `scenario_weights` for verifier-side consistency.

Verification policy (`sample_and_verify_linux.py`):
- Build eval cases using the same scenario set and weights (fallback to defaults, override from `meta.pkl` if present).
- Report both:
  - Accuracy by scenario
  - Accuracy by carry count
- Persist report output to `results/llm_output.txt`.

This policy is now the baseline for `4/5/6/7/8/10/20/50/80/100_digits_addition`.

## 7. Standard Workflow Commands

Example from `comp560-lamPham/4_digits_addition`:

```bash
cd /home/t1amp/comp560-LamPham/comp560-lamPham/4_digits_addition

# Prepare data
../../run.sh data/basic/prepare.py

# Train
../../run.sh -u ../common/train.py config/basic.py

# Verify
../../run.sh sample_and_verify_linux.py

# 5-run benchmark
./bench_run.sh
```

Run benchmark with a specific hyperopt config:

```bash
BENCH_CONFIG="$PWD/results/hyperopt/<run_id>/configs/trial_xxxx.py" ./bench_run.sh
```

## 8. Hyperparameter Tuning Workflow

Primary tuner:
- `comp560-lamPham/4_digits_addition/hyperopt_bot.py`

Example smoke run:

```bash
cd /home/t1amp/comp560-LamPham/comp560-lamPham/4_digits_addition
../../run.sh hyperopt_bot.py \
  --study-name smoke-$(date +%H%M%S) \
  --n-trials 3 \
  --timeout-min 20 \
  --max-iters 1200 \
  --eval-interval 100 \
  --eval-iters 2 \
  --log-interval 50 \
  --no-compile-trials
```

Outputs are stored under:
- `results/hyperopt/<run_id>/reports/trials.csv`
- `results/hyperopt/<run_id>/reports/champion.json`
- `results/hyperopt/<run_id>/configs/trial_xxxx.py`

## 9. Benchmarking Guidance

Use case recommendations:
- Fast local iteration: 1 run
- Candidate comparison: 3 runs
- Final champion decision: 5 runs (or more)

For final reporting, include:
- Mean time and standard deviation
- Mean accuracy and standard deviation
- Any outlier notes (thermal throttling, GPU contention, background load)

## 10. Git Hygiene and Generated Files

The repository ignores common generated artifacts such as:
- Local virtual environments (`.venv/`, `venv/`, `math_ai/`, `test_bot/`)
- Model outputs (`out/`, `out-*/`, `out_*/`)
- Checkpoints (`*.pt`, `*.ckpt`)
- Logs/results (`*.log`, `results/`, `wandb/`)
- Torch compile cache (`**/inductor_cache/`)

Do not commit runtime artifacts unless explicitly required for a release artifact workflow.

## 11. Troubleshooting Quick Reference

### `IsADirectoryError` when training
Common cause: config variable is empty and points to current directory.

Check:

```bash
echo "$CFG"
test -f "$CFG" && echo "CFG is valid"
```

### `Permission denied` on benchmark script

```bash
chmod +x bench_run.sh
```

### CUDA gatekeeper failure

```bash
./run.sh validate_env.py --profile core --require-cuda
```

### Unsure which interpreter is active

```bash
./run.sh --print-python
```

## 12. Developer and AI Checklist

Before running heavy jobs:
1. Verify interpreter path: `./run.sh --print-python`
2. Validate environment: `./run.sh validate_env.py --profile core --require-cuda`
3. Use `run.sh` for all training/verification commands
4. Record benchmark summary metrics (mean and std)

If setup conventions change, update this document first and notify the team.
