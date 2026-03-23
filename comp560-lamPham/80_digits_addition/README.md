# 80-Digit Addition Language Model Experiment

This folder trains and evaluates a 80-digit addition model using the shared trainer in ../common/train.py.

## Quick Start

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
- This dataset is sampled randomly (not carry-balanced) to keep generation tractable for large digit lengths.
