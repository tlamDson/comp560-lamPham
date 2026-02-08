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
├── sample_and_verify_linux.py    # Linux version (GPU)
├── sample_and_verify_windows.py  # Windows version (CPU)
├── config/
│   └── basic.py
├── data/
│   └── basic/
│       ├── prepare.py
│       ├── meta.pkl
│       ├── train.bin
│       └── val.bin
├── docs/
│   └── FINAL_REPORT.md
├── results/
│   ├── llm_output.txt
│   └── llm_output_cpu.txt
└── out/
```

---

## Setup

**1. Create Python virtual environment (from the root directory):**

```bash
# From the root comp560-LamPham directory
python -m venv venv
```

**2. Activate the virtual environment:**

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**3. Install dependencies:**

```bash
pip install torch numpy tqdm
```

**4. Prepare data:**

```bash
cd comp560-lamPham/arithmetic
python data/basic/prepare.py
```

**5. Check Hardware (Optional):**

To ensure GPU acceleration is available (recommended for speedrun):

```bash
python check_cuda.py
```

---

## Training

From the `arithmetic` directory:

**Linux:**
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

**Windows (PowerShell):**
```powershell
$env:NANOGPT_CONFIG="..\..\comp560-nanoGPT\configurator.py"
python -u ..\..\comp560-nanoGPT\train.py config\basic.py
```

---

## Sampling & Verification

After training, run the verification script to test the model's accuracy:

**Linux:**
```bash
python sample_and_verify_linux.py
```

**Windows:**
```powershell
python sample_and_verify_windows.py
```

### Manual Sampling (Optional)

**Linux:**
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=1 --max_new_tokens=5 --seed=42 --start="123+456="
```

**Windows (PowerShell):**
```powershell
$env:NANOGPT_CONFIG="..\..\comp560-nanoGPT\configurator.py"
python -u ..\..\comp560-nanoGPT\sample.py config\basic.py --num_samples=1 --max_new_tokens=5 --seed=42 --start="123+456="
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
