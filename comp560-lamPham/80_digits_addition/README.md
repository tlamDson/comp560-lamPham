# 80-Digit Addition Language Model Experiment

This folder trains and evaluates a 80-digit addition model using the shared trainer in ../common/train.py.

## Quick Start

Environment check (Torch/CUDA):
```bash
which python
python -c "import sys, torch, numpy; print(f'python: {sys.executable}'); print(f'torch: {torch.__version__}, numpy: {numpy.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Prepare data:
```bash
cd 80_digits_addition
python data/basic/prepare.py
```

Train:
```bash
time python -u ../common/train.py config/basic.py
```

Verify:
```bash
python sample_and_verify_linux.py
```

Benchmark:
```bash
chmod +x bench_run.sh
./bench_run.sh
```

## Notes

- Input format: 80-digit + 80-digit
- Output digits: 81
- Data preparation uses a data-centric scenario mix (stratified random, cascading carries, extreme imbalance, boundary/zero-heavy).
- Verification uses scenario-controlled test cases and reports accuracy by both scenario and carry count.
