# 100-Digit Addition Language Model Experiment

This folder trains and evaluates a 100-digit addition model using the shared trainer in ../common/train.py.

## Quick Start

Enter folder and activate conda:
```bash
cd /home/t1amp/comp560-LamPham/comp560-lamPham/100_digits_addition
conda activate base
```

Environment check (Torch/CUDA):
```bash
which python
python -c "import sys, torch, numpy; print(f'python: {sys.executable}'); print(f'torch: {torch.__version__}, numpy: {numpy.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Prepare data:
```bash
python data/basic/prepare.py
```

Train (NanoGPT, default in config/basic.py):
```bash
time python -u ../common/train.py config/basic.py
```

Train (Phi) via CLI override on the same config file:
```bash
time python -u ../common/train.py config/basic.py --model_family=phi --phi_model_source=Phi-3-mini-4k-instruct --out_dir=out_phi
```

Train (Phi) via dedicated config:
```bash
time python -u ../common/train.py config/phi_100_digits.py
```

Verify (NanoGPT checkpoint in out):
```bash
python sample_and_verify_linux.py
```

Verify (Phi checkpoint in out_phi):
```bash
OUT_DIR=out_phi python sample_and_verify_linux.py
```

If you trained Phi with CLI override but did NOT set `--out_dir=out_phi`, your checkpoint is in `out`:
```bash
OUT_DIR=out python sample_and_verify_linux.py
```

Benchmark:
```bash
chmod +x bench_run.sh
./bench_run.sh
```

## Notes

- Input format: 100-digit + 100-digit
- Output digits: 101
- Model switch in config/basic.py:
	- model_family='nanogpt' for NanoGPT
	- model_family='phi' for Phi adapter
- Data preparation uses a data-centric scenario mix (stratified random, cascading carries, extreme imbalance, boundary/zero-heavy).
- Verification uses scenario-controlled test cases and reports accuracy by both scenario and carry count.
