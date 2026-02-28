"""
Optimized Sample and Verify for 4-digit Addition.

This script loads the model ONCE and runs all predictions in-process,
avoiding the massive overhead of spawning a subprocess for each sample.

Performance comparison:
- Old (subprocess per sample): ~5-10 minutes for 100 samples
- New (in-process):            ~10-30 seconds for 100 samples
"""

import os
import sys
import random
import time
import pickle
from contextlib import nullcontext

import torch

# Add nanoGPT to path
NANOGPT_PATH = os.path.abspath("../../comp560-nanoGPT")
sys.path.insert(0, NANOGPT_PATH)

from model import GPTConfig, GPT

# ============================================================================
# Configuration
# ============================================================================
OUT_DIR = 'out'
CONFIG_FILE = 'config/basic.py'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

# Sampling parameters
MAX_NEW_TOKENS = 5  # 5-digit result for 4-digit addition
TEMPERATURE = 0.8
TOP_K = 200
SEED = 42

# Evaluation parameters
EVAL_PER_CARRY = 20  # 20 samples per carry count (0-4) = 100 total

# GPU monitoring interval (print stats every N samples)
GPU_MONITOR_INTERVAL = 20

# Device verification logging interval (Protocol 4.1)
DEVICE_LOG_INTERVAL = 100


# ============================================================================
# Helper Functions
# ============================================================================

def get_gpu_stats():
    """Get GPU memory and utilization stats."""
    if not torch.cuda.is_available():
        return "GPU: Not available"
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_memory:.2f}GB peak"


def count_carries(a, b):
    """Count carries for 4-digit addition."""
    carries = 0
    carry = 0
    for i in range(4):  # 4 digits
        digit_a = (a // (10 ** i)) % 10
        digit_b = (b // (10 ** i)) % 10
        if digit_a + digit_b + carry >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
    return carries


def format_prompt(a, b):
    """Format addition prompt: 1234+5678="""
    return f"{a:04d}+{b:04d}="


def generate_test_cases(seed, eval_per_carry):
    """Generate balanced test cases across carry counts."""
    random.seed(seed)
    cases = []
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    attempts = 0
    max_attempts = 100000
    
    while any(counts[c] < eval_per_carry for c in counts) and attempts < max_attempts:
        a = random.randint(1000, 9999)
        b = random.randint(1000, 9999)
        carries = count_carries(a, b)
        if counts[carries] < eval_per_carry:
            cases.append((a, b))
            counts[carries] += 1
        attempts += 1
    
    return cases, counts


# ============================================================================
# Model Loading
# ============================================================================

def load_model(out_dir, device, dtype):
    """Load the trained model from checkpoint with device verification."""
    print(f"Loading model from {out_dir}/ckpt.pt...")
    print(f"⚡ System Strategy: Active Device is [{device}]")
    print(f"Dtype: {dtype}")
    
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Build model from checkpoint config
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Load state dict (handle compiled model prefix)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # DEVICE GUARD: Verify model is on correct device (Protocol 2.2)
    model_device = next(model.parameters()).device
    if device == 'cuda' and model_device.type != 'cuda':
        print(f"❌ CRITICAL ERROR: Model is on {model_device}, not CUDA.")
        sys.exit(1)
    print(f"✅ Model loaded on device: {model_device}")
    
    # Get dataset info for encoding
    dataset = checkpoint['config'].get('dataset', 'basic')
    meta_path = os.path.join('data', dataset, 'meta.pkl')
    
    if os.path.exists(meta_path):
        print(f"Loading tokenizer from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(get_gpu_stats())
    
    return model, encode, decode, checkpoint


# ============================================================================
# Inference
# ============================================================================

def sample_single(model, encode, decode, prompt, device, dtype, max_new_tokens, temperature, top_k):
    """Generate tokens for a single prompt with device verification (Protocol 4.1)."""
    # Setup autocast context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Encode prompt
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # DEVICE GUARD: Verify input is on correct device (Protocol 4.1)
    if device_type == 'cuda' and x.device.type != 'cuda':
        print(f"❌ CRITICAL ERROR: Input data is on {x.device}, not CUDA.")
        sys.exit(1)
    
    # Generate
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # DEVICE GUARD: Verify output is on correct device (Protocol 4.1)
    if device_type == 'cuda' and y.device.type != 'cuda':
        print(f"❌ CRITICAL ERROR: Output tensor fell back to {y.device}.")
        sys.exit(1)
    
    # Decode full output
    output = decode(y[0].tolist())
    return output


def extract_prediction(output, prompt):
    """Extract the 5-digit sum from model output."""
    idx = output.find(prompt)
    if idx == -1:
        return None
    
    after = output[idx + len(prompt):]
    
    # Look for 5 consecutive digits
    import re
    match = re.search(r"(\d{5})", after)
    if not match:
        return None
    
    return match.group(1)


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("=" * 60)
    print("OPTIMIZED SAMPLE & VERIFY (4-DIGIT ADDITION)")
    print("=" * 60)
    print()
    
    # Set seeds
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Load model once
    start_load = time.time()
    model, encode, decode, checkpoint = load_model(OUT_DIR, DEVICE, DTYPE)
    load_time = time.time() - start_load
    print(f"Model load time: {load_time:.2f}s")
    print()
    
    # Generate test cases
    cases, counts = generate_test_cases(SEED, EVAL_PER_CARRY)
    total_cases = len(cases)
    print(f"Test cases: {total_cases}")
    print(f"Carry distribution: {counts}")
    print()
    
    # Track results
    correct = 0
    errors = []
    by_carry = {i: {"correct": 0, "total": 0} for i in range(5)}
    
    # Run evaluation
    print("Running evaluation...")
    print("-" * 60)
    
    start_eval = time.time()
    
    for i, (a, b) in enumerate(cases):
        prompt = format_prompt(a, b)
        carries = count_carries(a, b)
        by_carry[carries]["total"] += 1
        
        # Generate prediction
        output = sample_single(
            model, encode, decode, prompt,
            DEVICE, DTYPE, MAX_NEW_TOKENS, TEMPERATURE, TOP_K
        )
        
        predicted_str = extract_prediction(output, prompt)
        actual_sum = a + b
        
        # Check correctness
        if predicted_str is not None:
            try:
                predicted = int(predicted_str)
                if predicted == actual_sum:
                    correct += 1
                    by_carry[carries]["correct"] += 1
                else:
                    errors.append((a, b, predicted_str, actual_sum))
            except ValueError:
                errors.append((a, b, predicted_str, actual_sum))
        else:
            errors.append((a, b, None, actual_sum))
        
        # Progress and GPU stats
        if (i + 1) % GPU_MONITOR_INTERVAL == 0 or i == total_cases - 1:
            elapsed = time.time() - start_eval
            rate = (i + 1) / elapsed
            eta = (total_cases - i - 1) / rate if rate > 0 else 0
            print(f"Progress: {i+1}/{total_cases} ({100*(i+1)/total_cases:.0f}%) | "
                  f"Rate: {rate:.1f} samples/sec | ETA: {eta:.1f}s | "
                  f"Acc so far: {100*correct/(i+1):.1f}%")
            print(f"  {get_gpu_stats()}")
    
    eval_time = time.time() - start_eval
    
    # Print results
    print()
    print("=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Total cases: {total_cases}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {100 * correct / total_cases:.1f}%")
    print(f"Evaluation time: {eval_time:.2f}s ({total_cases/eval_time:.1f} samples/sec)")
    print()
    
    print("Accuracy by carry count:")
    for carry_count in sorted(by_carry.keys()):
        c_total = by_carry[carry_count]["total"]
        c_correct = by_carry[carry_count]["correct"]
        if c_total == 0:
            print(f"  carry {carry_count}: no samples")
        else:
            c_acc = 100 * c_correct / c_total
            print(f"  carry {carry_count}: {c_correct}/{c_total} ({c_acc:.1f}%)")
    
    if errors:
        print(f"\nErrors found: {len(errors)} (showing up to 10)")
        for a, b, pred_sum, actual_sum in errors[:10]:
            pred_str = pred_sum if pred_sum is not None else "None"
            print(f"  {a:04d}+{b:04d}={pred_str} should be {actual_sum:05d}")
    else:
        print("\nPERFECT! All predictions correct!")
    
    # Save report
    os.makedirs("results", exist_ok=True)
    report_lines = [
        "4-DIGIT ARITHMETIC OPTIMIZED EVAL REPORT",
        f"Total cases: {total_cases}",
        f"Correct: {correct}",
        f"Accuracy: {100 * correct / total_cases:.1f}%",
        f"Model load time: {load_time:.2f}s",
        f"Evaluation time: {eval_time:.2f}s",
        f"Throughput: {total_cases/eval_time:.1f} samples/sec",
        "",
        "Accuracy by carry count:",
    ]
    for carry_count in sorted(by_carry.keys()):
        c_total = by_carry[carry_count]["total"]
        c_correct = by_carry[carry_count]["correct"]
        if c_total == 0:
            report_lines.append(f"  carry {carry_count}: no samples")
        else:
            c_acc = 100 * c_correct / c_total
            report_lines.append(f"  carry {carry_count}: {c_correct}/{c_total} ({c_acc:.1f}%)")
    
    report_lines.append("")
    report_lines.append("Errors (up to 10):")
    for a, b, pred_sum, actual_sum in errors[:10]:
        pred_str = pred_sum if pred_sum is not None else "None"
        report_lines.append(f"  {a:04d}+{b:04d}={pred_str} should be {actual_sum:05d}")
    
    with open("results/llm_output.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")
    
    print(f"\nSaved report to results/llm_output.txt")
    print()
    print(f"Total time: {load_time + eval_time:.2f}s")
    print(get_gpu_stats())


if __name__ == "__main__":
    main()
