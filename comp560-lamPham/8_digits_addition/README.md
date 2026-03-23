# 8-Digit Addition Language Model Experiment

This folder trains and evaluates a 8-digit addition model using the shared trainer in `../common/train.py`.

## Quick Start

Prepare data:
```bash
cd 8_digits_addition
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

- Input format: 8-digit + 8-digit
- Output digits: 9
- Data uses reversed target digits to match current verifier and loss masking workflow.
