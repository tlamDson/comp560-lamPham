# comp560-LamPham: Arithmetic Model Experiments

This repository contains our end-to-end workflow for training and evaluating language models on algorithmic addition tasks (4 to 100 digits).

## What We Are Doing

We are building a reproducible experimentation pipeline to:
- Train models on synthetic digit-addition data.
- Compare configs and model families across digit lengths.
- Measure quality with consistent verification metrics.
- Benchmark performance with repeatable scripts.

Current standardization highlights:
- Data generation is standardized to a 70/10/10/10 scenario mix:
  - `stratified_random` (70%)
  - `cascading_carries` (10%)
  - `extreme_imbalance` (10%)
  - `boundary_zeros` (10%)
- Verification reports:
  - Overall accuracy
  - Accuracy by scenario
  - Accuracy by carry count

## Repository Layout

- `comp560-lamPham/`
  - Main experiment code and digit-specific folders (`4_digits_addition` to `100_digits_addition`)
  - Shared training entrypoint in `common/train.py`
- `comp560-nanoGPT/`
  - NanoGPT-based model and training components
- `Phi-3-mini-4k-instruct/`
  - Local Phi model assets for Phi workflows
- `scripts/`
  - Lock/install helpers for reproducible environments
- `run.sh`
  - Canonical wrapper to run Python commands consistently
- `validate_env.py`
  - Environment gatekeeper (interpreter, imports, CUDA, prerequisites)

## Quick Start

From repository root:

```bash
cp .env.example .env
chmod +x run.sh
./run.sh --print-python
./run.sh validate_env.py --profile core --require-cuda
```

## Typical Workflow (Example: 4 Digits)

```bash
cd comp560-lamPham/4_digits_addition
../../run.sh data/basic/prepare.py
../../run.sh -u ../common/train.py config/basic.py
../../run.sh sample_and_verify_linux.py
./bench_run.sh
```

## Where to Read More

- `developer_set_up.md`:
  - Full English setup guide
  - Runtime policy and dependency workflow
  - Benchmarking and troubleshooting details
- Folder READMEs under `comp560-lamPham/*_digits_addition/`:
  - Task-specific notes and run details

## Notes

- Use `run.sh` for normal project workflows to avoid interpreter drift.
- Avoid committing runtime artifacts (logs, checkpoints, local envs).
