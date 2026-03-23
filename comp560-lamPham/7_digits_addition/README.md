# 7-Digit Addition Language Model Experiment

This folder trains and evaluates a 7-digit addition model using the shared trainer in `../common/train.py`.

## Quick Start

Prepare data:
```bash
cd 7_digits_addition
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

- Input format: 7-digit + 7-digit
- Output digits: 8
- Data uses reversed target digits to match current verifier and loss masking workflow.
