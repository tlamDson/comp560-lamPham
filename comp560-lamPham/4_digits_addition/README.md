# 4-Digit Addition Language Model Experiment

This experiment trains a character-level transformer language model to perform 4-digit addition with **100% accuracy in ~15 seconds** using optimized training configurations.

---

## Data Format

The model learns from a stream of addition examples in the format:

```
1234+5678=06912
9050+0990=10040
0001+0002=00003
```

Each example is exactly 20 characters (4+1+4+1+5+padding), designed to teach the model carry propagation in multi-digit arithmetic.

---

## Directory Structure

```
4_digits_addition/
├── README.md                      # This file
├── bench_run.sh                   # Automated benchmarking script
├── calculate_mfu.py               # Model FLOPs utilization calculator
├── check_cuda.py                  # CUDA availability checker
├── sample_and_verify_linux.py     # GPU-based sampling & verification
├── sample_and_verify_windows.py   # Windows version (CPU)
├── sample_cpu.py                  # CPU fallback
├── training.log                   # Training output log
├── config/
│   └── basic.py                   # ⭐ OPTIMIZED CONFIG (100% accuracy)
├── data/
│   └── basic/
│       ├── prepare.py             # Dataset generator
│       ├── meta.pkl               # Character vocabulary
│       ├── train.bin              # Training data (250k examples)
│       └── val.bin                # Validation data
├── docs/
│   └── FINAL_REPORT.md
├── results/
│   ├── llm_output.txt             # GPU output logs
│   └── llm_output_cpu.txt         # CPU output logs
├── out/                           # Model checkpoints
│   └── ckpt.pt                    # Latest trained model
└── wandb/                         # Weights & Biases logs (optional)
```

---

## Setup

### 1. Environment Setup

**Option A: Use Conda environment (recommended for GPU)**

```bash
# Verify packages are available
python -c "import torch, numpy; print(f'torch: {torch.__version__}, numpy: {numpy.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Option B: Create Python virtual environment**

```bash
# From the root comp560-LamPham directory
python -m venv venv
source venv/bin/activate
pip install torch numpy tqdm
```

> **Warning:** Don't mix venv and Conda! If you see `(base)` in your prompt, you're using Conda - no need to activate venv.

### 2. Prepare Dataset

```bash
cd comp560-lamPham/4_digits_addition
python data/basic/prepare.py
```

This generates 250k training examples and validation data.

### 3. Verify CUDA (Optional but Recommended)

```bash
python check_cuda.py
```

Confirms GPU acceleration is available for fast training.

---

## Training

### Quick Start (Recommended)

**With output to screen** (see training progress):

```bash
time python -u ../common/train.py config/basic.py
```

**Silent mode** (redirect to `/dev/null` - saves CPU time by not printing to screen):

```bash
time python -u ../common/train.py config/basic.py > /dev/null
```

> **Note:** Using `> /dev/null` eliminates console I/O overhead, resulting in faster execution. This is especially useful for CPU-bound systems or when running benchmarks.

### Expected Performance

With the optimized configuration in `config/basic.py`:

- **Time**: ~15 seconds (on modern GPU)
- **Accuracy**: 100% (model auto-stops at >99% validation accuracy)
- **Final val_acc**: ≥99%

---

## Benchmarking (Multiple Runs)

To run multiple training sessions and compute average performance metrics:

```bash
chmod +x bench_run.sh
./bench_run.sh
```

**Customizing number of runs:**

Edit `bench_run.sh` and modify:

```bash
NUM_RUNS=5  # Change this to desired number of runs
```

The script will:
1. Run training `NUM_RUNS` times
2. After each training, run `sample_and_verify_linux.py`
3. Extract: Real Time, Final MFU (%), and Accuracy (%)
4. Output a summary table with Mean and Standard Deviation

Example output:

```
Trial  Real Time    MFU (%)    Accuracy (%)
-----  ----------  ---------  --------------
1      15.23s      45.67%     100.00%
2      14.98s      46.12%     100.00%
...
Mean   15.05s      45.92%     100.00%
StdDev  0.18s       0.23%       0.00%
```

---

## GPU Monitoring Commands

### Real-time GPU Utilization

**Watch GPU usage every 1 second:**

```bash
nvidia-smi -l 1
```

Shows GPU memory, utilization %, temperature, and power consumption.

**Watch SM (Streaming Multiprocessor) activity and detailed metrics:**

```bash
nvidia-smi dmon
```

Displays:
- `sm`: SM utilization (%)
- `mem`: Memory utilization (%)
- `enc`: Encoder utilization
- `dec`: Decoder utilization
- `pwr`: Power usage (W)

**Advanced monitoring with specific metrics:**

```bash
nvidia-smi dmon -s pucvmet
```

Options:
- `p`: Power usage
- `u`: SM utilization
- `c`: Clock frequencies
- `v`: Violations (power/thermal throttling)
- `m`: Memory usage
- `e`: ECC errors
- `t`: Temperature

**Watch GPU during training:**

```bash
# In terminal 1
nvidia-smi -l 1

# In terminal 2
time python -u ../common/train.py config/basic.py
```

---

## Sampling & Verification

After training, test the model's accuracy on 100 test cases:

```bash
python sample_and_verify_linux.py
```

This script:
- Loads the trained model from `out/ckpt.pt`
- Generates 100 predictions (20 per carry count: 0, 1, 2, 3, 4)
- Reports overall accuracy and breakdown by carry complexity

### Manual Sampling (Optional)

Test individual predictions:

```bash
python -u ../../comp560-nanoGPT/sample.py config/basic.py --num_samples=1 --max_new_tokens=5 --seed=42 --start="1234+5678="
```

---

## Optimized Configuration Highlights

The current `config/basic.py` achieves **100% accuracy in ~15 seconds** with these optimizations:

### Model Architecture (Compact & Efficient)

```python
n_layer = 4         # 4 transformer layers
n_head = 8          # 8 attention heads
n_embd = 64         # 64 embedding dimensions
block_size = 20     # Sequence length (4+1+4+1+5+pad)
batch_size = 1024   # Large batch for stable gradients
```

### Training Hyperparameters

```python
learning_rate = 6e-3     # High LR for fast convergence
max_iters = 5000         # Max iterations (early stop usually triggers)
warmup_iters = 100       # Quick warmup
early_stop_loss = 1.12   # Loss-based early stopping
dropout = 0.0            # No dropout (small model, large dataset)
weight_decay = 0.0       # Allow sharp weights for carry calculation
```

### Key Features

- **answer_only_loss = True**: Loss computed only on the answer portion (5 digits), not the prompt
- **Auto-stop at 99% accuracy**: Training stops when validation accuracy exceeds 99%
- **Validation monitoring**: Prints `val_acc` every 200 iterations

---

## Key Changes in train.py (comp560-lamPham/common)

### 1. Print Validation Accuracy

**Modified:** Shared trainer in `../common/train.py`

```python
print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val acc {losses['val_acc']*100:.2f}%")
```

Now displays validation accuracy percentage during training, making it easy to monitor convergence.

### 2. Auto-Stop at 99% Accuracy

**Added:** Shared trainer policy in `../common/train.py`

```python
early_stop_acc = 0.99  # 99% threshold
if losses['val_acc'] >= early_stop_acc:
    print(f"Early stopping triggered! val_acc {losses['val_acc']*100:.2f}% >= threshold {early_stop_acc*100}%")
    break
```

Training automatically stops when validation accuracy hits ≥99%, saving time and preventing overfitting.

### 3. Log Accuracy to Weights & Biases

**Added:** Shared trainer logging in `../common/train.py`

```python
"val/acc": losses['val_acc'],  # Log accuracy to wandb
```

Enables tracking accuracy curves in wandb dashboards.

---

## Performance Comparison

| Aspect | 3-digit | 4-digit (This Experiment) |
|--------|---------|---------------------------|
| Number range | 100-999 | 1000-9999 |
| Result format | 4 digits | 5 digits |
| Max carries | 3 | 4 |
| block_size | 16 | 20 |
| Dataset size | 240k examples | 250k examples |
| Optimized Time | ~12s | ~15s |
| Optimized Accuracy | 100% | 100% |

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce `batch_size` in `config/basic.py`:

```python
batch_size = 512  # Down from 1024
```

### Issue: Slow Training on CPU

**Solution:** Use silent mode to reduce I/O overhead:

```bash
time python -u ../common/train.py config/basic.py > /dev/null
```

### Issue: Model Not Reaching 99% Accuracy

**Solution:** Check that you're using the optimized `config/basic.py` with:
- `learning_rate = 6e-3`
- `answer_only_loss = True`
- `max_iters = 5000`

---

## Results Summary

Using the optimized configuration:

✅ **Training Time**: ~15 seconds (NVIDIA GPU)  
✅ **Validation Accuracy**: 100% (auto-stops at >99%)  
✅ **Test Accuracy**: 100% (100/100 correct predictions)  
✅ **Convergence**: Stable, no overfitting  

---

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Original 3-digit addition experiment in `/comp560-lamPham/arithmetic/`
- Training modifications in `/comp560-lamPham/common/train.py`
