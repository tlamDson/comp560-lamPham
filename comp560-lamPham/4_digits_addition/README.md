# Arithmetic (4-digit addition) Experiment

This experiment trains a character-level language model to perform 4-digit addition.

---

## Data Format

The model learns from a stream of addition examples:

- 1234+5678=06912
- 9050+0990=10040
- ...

---

## Directory Structure

```
4_digits_addition/
├── README.md
├── check_cuda.py
├── sample_and_verify_linux.py    # Linux version (GPU)
├── sample_and_verify_windows.py  # Windows version (CPU)
├── sample_cpu.py                 # CPU fallback
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
cd comp560-lamPham/4_digits_addition
python data/basic/prepare.py
```

**5. Check Hardware (Optional):**

To ensure GPU acceleration is available (recommended for speedrun):

```bash
python check_cuda.py
```

---

## Training

From the `4_digits_addition` directory:

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
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=1 --max_new_tokens=5 --seed=42 --start="1234+5678="
```

**Windows (PowerShell):**
```powershell
$env:NANOGPT_CONFIG="..\..\comp560-nanoGPT\configurator.py"
python -u ..\..\comp560-nanoGPT\sample.py config\basic.py --num_samples=1 --max_new_tokens=5 --seed=42 --start="1234+5678="
```

---

## Key Differences from 3-digit Addition

| Aspect | 3-digit | 4-digit |
|--------|---------|---------|
| Number range | 100-999 | 1000-9999 |
| Result format | 4 digits | 5 digits |
| Max carries | 3 | 4 |
| block_size | 16 | 20 |
| max_iters | 6000 | 8000 |
| Dataset size | 240k examples | 250k examples |

---

## Experiment Log

_(To be filled after running experiments)_

### Run 1: Baseline

**Purpose**: Establish baseline performance for 4-digit addition

**Config**: max_iters=8000, device=auto-detect

**Results**:

- **Time**: TBD
- **Accuracy**: TBD%

---

## Expected Challenges

Compared to 3-digit addition:

1. **Larger search space**: 4-digit × 4-digit = 81M possible combinations (vs 810k for 3-digit)
2. **More carries**: Up to 4 carries possible (vs 3 for 3-digit)
3. **Longer sequences**: Each example is 16 characters (vs 13 for 3-digit)

---

## Conclusion

_(To be filled after experiments)_
