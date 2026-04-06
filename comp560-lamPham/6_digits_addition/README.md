# 6-Digit Addition Language Model Experiment

This folder trains and evaluates a 6-digit addition model using the shared trainer in `../common/train.py`.

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

- Input format: 6-digit + 6-digit
- Output digits: 7
- Data uses reversed target digits to match current verifier and loss masking workflow.
