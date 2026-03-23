"""
Sample and verify 8-digit addition with reversed targets.
"""
import os
import sys
import random
import time
import pickle
import re
from contextlib import nullcontext

import torch

NANOGPT_PATH = os.path.abspath("../../comp560-nanoGPT")
sys.path.insert(0, NANOGPT_PATH)
from model import GPTConfig, GPT

NUM_DIGITS = 8
RESULT_DIGITS = NUM_DIGITS + 1
MIN_NUM = 10000000
MAX_NUM = 99999999

OUT_DIR = 'out'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
MAX_NEW_TOKENS = RESULT_DIGITS
TEMPERATURE = 0.8
TOP_K = 200
SEED = 42
EVAL_PER_CARRY = 20


def count_carries(a, b):
    carries = 0
    carry = 0
    for i in range(NUM_DIGITS):
        digit_a = (a // (10 ** i)) % 10
        digit_b = (b // (10 ** i)) % 10
        if digit_a + digit_b + carry >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
    return carries


def format_prompt(a, b):
    return f"{a:0{NUM_DIGITS}d}+{b:0{NUM_DIGITS}d}="


def load_model(out_dir, device):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    dataset = checkpoint['config'].get('dataset', 'basic')
    meta_path = os.path.join('data', dataset, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return model, encode, decode


def sample_single(model, encode, decode, prompt, device, dtype):
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        with ctx:
            y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
    return decode(y[0].tolist())


def extract_prediction(output, prompt):
    idx = output.find(prompt)
    if idx == -1:
        return None
    after = output[idx + len(prompt):]
    match = re.search(r"(\d{%d})" % RESULT_DIGITS, after)
    if not match:
        return None
    return match.group(1)[::-1]


def generate_test_cases(seed, per_carry):
    random.seed(seed)
    cases = []
    counts = {c: 0 for c in range(NUM_DIGITS + 1)}
    while any(counts[c] < per_carry for c in counts):
        a = random.randint(MIN_NUM, MAX_NUM)
        b = random.randint(MIN_NUM, MAX_NUM)
        carries = count_carries(a, b)
        if counts[carries] < per_carry:
            counts[carries] += 1
            cases.append((a, b))
    return cases, counts


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print(f"{NUM_DIGITS}-DIGIT SAMPLE & VERIFY")
    model, encode, decode = load_model(OUT_DIR, DEVICE)
    cases, counts = generate_test_cases(SEED, EVAL_PER_CARRY)
    print(f"carry distribution: {counts}")

    by_carry = {c: {'correct': 0, 'total': 0} for c in range(NUM_DIGITS + 1)}
    correct = 0
    errors = []

    start = time.time()
    for a, b in cases:
        prompt = format_prompt(a, b)
        carries = count_carries(a, b)
        by_carry[carries]['total'] += 1
        output = sample_single(model, encode, decode, prompt, DEVICE, DTYPE)
        predicted_str = extract_prediction(output, prompt)
        actual_sum = a + b
        if predicted_str is not None and predicted_str.isdigit() and int(predicted_str) == actual_sum:
            correct += 1
            by_carry[carries]['correct'] += 1
        else:
            errors.append((a, b, predicted_str, actual_sum))

    total = len(cases)
    elapsed = time.time() - start
    print(f"Accuracy: {100 * correct / total:.1f}%")
    print(f"Evaluation time: {elapsed:.2f}s")

    for carry_count in sorted(by_carry.keys()):
        c_total = by_carry[carry_count]['total']
        c_correct = by_carry[carry_count]['correct']
        if c_total == 0:
            print(f"  carry {carry_count}: no samples")
        else:
            print(f"  carry {carry_count}: {c_correct}/{c_total} ({100*c_correct/c_total:.1f}%)")

    os.makedirs('results', exist_ok=True)
    with open('results/llm_output.txt', 'w') as f:
        f.write(f"8-DIGIT EVAL REPORT\n")
        f.write(f"Total cases: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {100 * correct / total:.1f}%\n")
        f.write(f"Throughput: {total/elapsed:.1f} samples/sec\n\n")
        f.write("Accuracy by carry count:\n")
        for carry_count in sorted(by_carry.keys()):
            c_total = by_carry[carry_count]['total']
            c_correct = by_carry[carry_count]['correct']
            if c_total == 0:
                f.write(f"  carry {carry_count}: no samples\n")
            else:
                f.write(f"  carry {carry_count}: {c_correct}/{c_total} ({100*c_correct/c_total:.1f}%)\n")
        f.write("\nSample errors (up to 10):\n")
        for a, b, pred_sum, actual_sum in errors[:10]:
            pred_str = pred_sum if pred_sum is not None else 'None'
            f.write(f"  {a:0{NUM_DIGITS}d}+{b:0{NUM_DIGITS}d}={pred_str} should be {actual_sum:0{RESULT_DIGITS}d}\n")


if __name__ == '__main__':
    main()
