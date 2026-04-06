# 50-Digit Addition Language Model Experiment

This folder trains and evaluates a 50-digit addition model using the shared trainer in ../common/train.py.

## Quick Start

Environment check (recommended wrapper + gatekeeper):
```bash
# From repository root (one-time setup)
cp .env.example .env
chmod +x run.sh
./run.sh validate_env.py --profile core --require-cuda
```

Prepare data:
```bash
../../run.sh data/basic/prepare.py
```

Train:
```bash
time ../../run.sh -u ../common/train.py config/basic.py
```

Verify:
```bash
../../run.sh sample_and_verify_linux.py
```

Benchmark:
```bash
chmod +x bench_run.sh
./bench_run.sh
```

## Notes

- Input format: 50-digit + 50-digit
- Output digits: 51
- Data preparation is carry-balanced across all carry counts (0 through 50).
- Verification uses carry-controlled test cases so every carry count (0 through 50) is evaluated.
