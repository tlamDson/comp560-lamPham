"""
Sample and automatically verify 3-digit addition results.
This script loads the model ONCE and runs all predictions on GPU efficiently.

Linux version - uses GPU for fast inference.
"""
import os
import sys
import random
import pickle

# Add nanoGPT to path
NANOGPT_PATH = os.path.abspath("../../comp560-nanoGPT")
sys.path.insert(0, NANOGPT_PATH)

import torch
from model import GPT, GPTConfig

# Configuration
CHECKPOINT_PATH = "out/ckpt.pt"
EVAL_PER_CARRY = 20
SEED = 42
TEMPERATURE = 1e-8  # Near-greedy decoding (0 would cause div by zero)
TOP_K = 1  # Only top token
MAX_NEW_TOKENS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

print(f"AUTOMATED SAMPLE & VERIFY (ADDITION) - Linux")
print(f"Device: {DEVICE}, Dtype: {DTYPE}")

def count_carries(a, b):
    carries = 0
    carry = 0
    for i in range(3):
        digit_a = (a // (10 ** i)) % 10
        digit_b = (b // (10 ** i)) % 10
        if digit_a + digit_b + carry >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
    return carries

def load_model():
    """Load the trained model from checkpoint."""
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Load meta for encoding/decoding
    meta_path = os.path.join('data', 'basic', 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Create model
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(DEVICE)
    
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, encode, decode

@torch.no_grad()
def predict(model, encode, decode, prompt):
    """Generate prediction for a single prompt."""
    x = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    with torch.amp.autocast(device_type='cuda', dtype=DTYPE):
        y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
    
    output = decode(y[0].tolist())
    # Extract the 4-digit result after the prompt
    result = output[len(prompt):len(prompt)+4]
    return result if len(result) == 4 and result.isdigit() else None

def main():
    # Load model once
    model, encode, decode = load_model()
    
    # Generate test cases
    random.seed(SEED)
    cases = []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    while any(counts[c] < EVAL_PER_CARRY for c in counts):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        carries = count_carries(a, b)
        if counts[carries] < EVAL_PER_CARRY:
            cases.append((a, b))
            counts[carries] += 1
    
    print(f"\nRunning prompt-based evaluation...")
    print(f"Evaluation cases: {len(cases)} (carry distribution: {counts})\n")
    
    # Run predictions
    correct = 0
    total = 0
    errors = []
    by_carry = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}, 
                2: {"correct": 0, "total": 0}, 3: {"correct": 0, "total": 0}}
    
    for a, b in cases:
        prompt = f"{a:03d}+{b:03d}="
        predicted_str = predict(model, encode, decode, prompt)
        actual_sum = a + b
        carries = count_carries(a, b)
        by_carry[carries]["total"] += 1
        total += 1
        
        if predicted_str is None:
            errors.append((a, b, None, actual_sum))
            continue
        
        try:
            predicted_sum = int(predicted_str)
        except ValueError:
            errors.append((a, b, predicted_str, actual_sum))
            continue
        
        if predicted_sum == actual_sum:
            correct += 1
            by_carry[carries]["correct"] += 1
        else:
            errors.append((a, b, predicted_str, actual_sum))
    
    # Generate report
    report_lines = []
    report_lines.append("ARITHMETIC PROMPT EVAL REPORT")
    report_lines.append(f"Total cases: {total}")
    report_lines.append(f"Correct: {correct}")
    report_lines.append(f"Accuracy: {(correct / total) * 100:.1f}%")
    report_lines.append("")
    report_lines.append("Accuracy by carry count:")
    for carry_count in sorted(by_carry.keys()):
        c_total = by_carry[carry_count]["total"]
        c_correct = by_carry[carry_count]["correct"]
        if c_total == 0:
            report_lines.append(f"  carry {carry_count}: no samples")
        else:
            c_acc = (c_correct / c_total) * 100
            report_lines.append(f"  carry {carry_count}: {c_correct}/{c_total} ({c_acc:.1f}%)")
    report_lines.append("")
    report_lines.append("Sample errors (up to 10):")
    for a, b, pred_sum, actual_sum in errors[:10]:
        pred_str = pred_sum if pred_sum is not None else "None"
        report_lines.append(f"  {a:03d}+{b:03d}={pred_str} should be {actual_sum:04d}")
    
    os.makedirs("results", exist_ok=True)
    with open("results/llm_output.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")
    
    # Print results
    print("VERIFICATION RESULTS")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {(correct / total) * 100:.1f}%")
    print("\nAccuracy by carry count:")
    for carry_count in sorted(by_carry.keys()):
        c_total = by_carry[carry_count]["total"]
        c_correct = by_carry[carry_count]["correct"]
        if c_total == 0:
            print(f"  carry {carry_count}: no samples")
        else:
            c_acc = (c_correct / c_total) * 100
            print(f"  carry {carry_count}: {c_correct}/{c_total} ({c_acc:.1f}%)")
    
    if errors:
        print(f"\nErrors found: {len(errors)} (showing up to 10)")
        for a, b, pred_sum, actual_sum in errors[:10]:
            pred_str = pred_sum if pred_sum is not None else "None"
            print(f"  {a:03d}+{b:03d}={pred_str} should be {actual_sum:04d}")
    else:
        print("\nPERFECT! All predictions correct!")
    
    print("\nSaved report to results/llm_output.txt")

if __name__ == "__main__":
    main()
