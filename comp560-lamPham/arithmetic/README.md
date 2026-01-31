# Arithmetic (3-digit addition) Experiment

This experiment trains a character-level language model to perform 3-digit addition.

---

## Data Format

The model learns from a stream of addition examples:

- 123+456=0579
- 905+099=1004
- ...

---

## Directory Structure

```
arithmetic/
├── README.md
├── check_cuda.py
├── config/
│   └── basic.py
├── data/
│   └── basic/
│       ├── prepare.py
│       ├── meta.pkl
│       ├── train.bin
│       └── val.bin
├── docs/
│   ├── FINAL_REPORT.md
├── results/
│   ├── llm_output.txt
│   └── llm_output_cpu.txt
└── out/
```

---

## Setup

**1. Create Python virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

**2. Install dependencies:**

```bash
pip install torch numpy tqdm
```

**3. Prepare data:**

```bash
cd arithmetic
python data/basic/prepare.py
```

**4. Check Hardware (Optional):**

To ensure GPU acceleration is available (recommended for speedrun):

```bash
python check_cuda.py
```

---

## Training

From the `arithmetic` directory:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

---

## Sampling

After training:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=1 --max_new_tokens=5 --seed=42 --start="123+456="
```

---

## Experiment Log

### Run 1: Baseline (CPU)

**Purpose**: Establish baseline performance

**Config**: max_iters=3000, device=cpu, compile=False

**Results**:

- **Time**: ~6 minutes
- **Accuracy**: 6.4%
- **Issues**: Variable length outputs and unconditional sampling caused poor performance.

### Run 2: Improved Baseline (CPU)

**Purpose**: Fix data formatting and evaluation

**Config**: max_iters=8000, device=cpu, fixed 4-digit output format

**Results**:

- **Time**: ~6 minutes
- **Accuracy**: 100% (80/80 correct)
- **Insight**: Deterministic data formatting is crucial.

### Run 3: Speedrun (GPU)

**Purpose**: Optimize for speed on RTX 4060

**Config**: max_iters=6000, device=cuda, dtype=bfloat16 (bf16), learning_rate=3e-3

**Final Setup Note**:
The configuration (`config/basic.py`) automatically detects CUDA availability.

- **GPU Mode**: Uses `device='cuda'` and `dtype='bfloat16'` for maximum speed (~3 mins).
- **CPU Mode**: Fallback if no GPU detected (~6 mins).

**Results**:

- **All predictions were accurate!**
- **Time**: 3 minutes 10 seconds
- **Accuracy**: 100%
- **Speedup**: ~1.9x faster than CPU baseline

**Training Loss Graph (from Wandb):**

_(Graph placeholder or link if available)_

---

## Conclusion

**What worked:**

- Fixed output format (always 4 digits) dramatically improved accuracy
- GPU acceleration with `bfloat16` reduced training time significantly
- CPU fallback for sampling ensured stability on Windows

**What didn't work:**

- `torch.compile` is not supported on Windows
- GPU sampling was unstable on Windows (crashes)

**Changes made:**

- Implemented `sample_cpu.py` for reliable evaluation
- Tuned learning rate and iterations for optimal convergence speed
