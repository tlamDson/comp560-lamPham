"""
Sample and verify 6-digit addition with reversed targets.
"""

import os
import pickle
import random
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext

import torch

NANOGPT_PATH = os.path.abspath("../../comp560-nanoGPT")
sys.path.insert(0, NANOGPT_PATH)
from model import GPT, GPTConfig

NUM_DIGITS = 6
RESULT_DIGITS = NUM_DIGITS + 1
OUT_DIR = os.environ.get("OUT_DIR", "out")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
MAX_NEW_TOKENS = RESULT_DIGITS
TEMPERATURE = 0.8
TOP_K = 200
SEED = 42
TOTAL_EVAL_CASES = 140
SCENARIO_WEIGHTS = {
    "stratified_random": 0.70,
    "cascading_carries": 0.10,
    "extreme_imbalance": 0.10,
    "boundary_zeros": 0.10,
}


def random_with_length(num_digits):
    if num_digits < 1 or num_digits > NUM_DIGITS:
        raise ValueError(f"num_digits must be in [1, {NUM_DIGITS}]")
    if num_digits == 1:
        return random.randint(0, 9)
    low = 10 ** (num_digits - 1)
    high = (10 ** num_digits) - 1
    return random.randint(low, high)


def stratified_random_pair():
    len_a = random.randint(1, NUM_DIGITS)
    len_b = random.randint(1, NUM_DIGITS)
    return random_with_length(len_a), random_with_length(len_b)


def cascading_carry_pair():
    chain_min = max(2, NUM_DIGITS // 3)
    chain_len = random.randint(chain_min, NUM_DIGITS)
    high_len = random.randint(chain_len, NUM_DIGITS)

    high = random_with_length(high_len)
    high_str = f"{high:0{high_len}d}"
    left = high_str[:-chain_len] if chain_len < len(high_str) else ""
    tail = "9" * chain_len
    a = int((left + tail).zfill(high_len))

    b = random.randint(1, 9)
    if random.random() < 0.35:
        max_extra = min(6, NUM_DIGITS - 1)
        if max_extra >= 2:
            extra_len = random.randint(2, max_extra)
            b = random_with_length(extra_len)
            b = (b // 10) * 10 + random.randint(1, 9)

    return min(a, 10**NUM_DIGITS - 1), min(b, 10**NUM_DIGITS - 1)


def extreme_imbalance_pair():
    len_big_min = max(2, int(round(NUM_DIGITS * 0.7)))
    len_big = random.randint(len_big_min, NUM_DIGITS)
    len_small_max = max(1, min(3, NUM_DIGITS - 1))
    len_small = random.randint(1, len_small_max)

    big = random_with_length(len_big)
    if random.random() < 0.7:
        small = random.choice([0, 1])
    else:
        small = random_with_length(len_small)

    return (big, small) if random.random() < 0.5 else (small, big)


def boundary_zeros_pair():
    mode = random.random()

    if mode < 0.4:
        a = random.randint(5 * 10 ** (NUM_DIGITS - 1), 10**NUM_DIGITS - 1)
        b = random.randint(5 * 10 ** (NUM_DIGITS - 1), 10**NUM_DIGITS - 1)
        return a, b

    if mode < 0.8:
        def zero_heavy(length):
            digits = ["0"] * length
            non_zero_count = random.randint(1, max(1, length // 8))
            picks = random.sample(range(length), non_zero_count)
            for idx in picks:
                digits[idx] = str(random.randint(1, 9))
            if digits[0] == "0":
                digits[0] = str(random.randint(1, 9))
            return int("".join(digits))

        len_a = random.randint(1, NUM_DIGITS)
        len_b = random.randint(1, NUM_DIGITS)
        return zero_heavy(len_a), zero_heavy(len_b)

    a = 10**NUM_DIGITS - random.randint(1, 99)
    b = random.choice([0, 1, 2, 5, 9])
    return a, b


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


def scenario_counts(total_examples, scenario_weights):
    names = list(scenario_weights.keys())
    counts = {name: int(total_examples * scenario_weights[name]) for name in names}
    assigned = sum(counts.values())
    remainder = total_examples - assigned
    for name in sorted(names, key=lambda n: scenario_weights[n], reverse=True)[:remainder]:
        counts[name] += 1
    return counts


def canonical_pair(a, b):
    return (a, b) if a <= b else (b, a)


def build_eval_cases(total_examples, scenario_weights):
    generators = {
        "stratified_random": stratified_random_pair,
        "cascading_carries": cascading_carry_pair,
        "extreme_imbalance": extreme_imbalance_pair,
        "boundary_zeros": boundary_zeros_pair,
    }

    counts = scenario_counts(total_examples, scenario_weights)
    cases = []
    seen = set()

    for scenario, target in counts.items():
        generator = generators[scenario]
        accepted = 0
        attempts = 0
        while accepted < target:
            attempts += 1
            if attempts > target * 200:
                raise RuntimeError(f"Too many duplicate attempts in scenario {scenario}")

            a, b = generator()
            if a < 0 or b < 0 or a >= 10**NUM_DIGITS or b >= 10**NUM_DIGITS:
                continue

            key = canonical_pair(a, b)
            if key in seen:
                continue
            seen.add(key)
            cases.append((a, b, scenario))
            accepted += 1

    random.shuffle(cases)
    return cases


def format_prompt(a, b):
    return f"{a:0{NUM_DIGITS}d}+{b:0{NUM_DIGITS}d}="


def load_model(out_dir, device):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    dataset = checkpoint["config"].get("dataset", "basic")
    meta_path = os.path.join("data", dataset, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    return model, encode, decode


def sample_single(model, encode, decode, prompt, device, dtype):
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        with ctx:
            y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
    return decode(y[0].tolist())


def extract_prediction(output, prompt):
    idx = output.find(prompt)
    if idx == -1:
        return None
    after = output[idx + len(prompt) :]
    match = re.search(r"(\d{%d})" % RESULT_DIGITS, after)
    if not match:
        return None
    return match.group(1)[::-1]


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print(f"{NUM_DIGITS}-DIGIT SAMPLE & VERIFY")
    model, encode, decode = load_model(OUT_DIR, DEVICE)

    scenario_weights = dict(SCENARIO_WEIGHTS)
    meta_path = os.path.join("data", "basic", "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if isinstance(meta.get("scenario_weights"), dict):
            scenario_weights = meta["scenario_weights"]

    eval_cases = build_eval_cases(TOTAL_EVAL_CASES, scenario_weights)

    by_carry = {c: {"correct": 0, "total": 0} for c in range(NUM_DIGITS + 1)}
    by_scenario = {name: {"correct": 0, "total": 0} for name in scenario_weights}
    scenario_eval_hist = Counter()
    correct = 0
    errors = []

    start = time.time()
    for a, b, scenario in eval_cases:
        prompt = format_prompt(a, b)
        carries = count_carries(a, b)

        by_carry[carries]["total"] += 1
        by_scenario[scenario]["total"] += 1
        scenario_eval_hist[scenario] += 1

        output = sample_single(model, encode, decode, prompt, DEVICE, DTYPE)
        predicted_str = extract_prediction(output, prompt)
        actual_sum = a + b

        if predicted_str is not None and predicted_str.isdigit() and int(predicted_str) == actual_sum:
            correct += 1
            by_carry[carries]["correct"] += 1
            by_scenario[scenario]["correct"] += 1
        else:
            errors.append((scenario, a, b, predicted_str, actual_sum))

    total = len(eval_cases)
    elapsed = time.time() - start

    print(f"Accuracy: {100 * correct / total:.1f}%")
    print(f"Evaluation time: {elapsed:.2f}s")

    print("Scenario mix in evaluation:")
    for scenario in scenario_weights:
        print(f"  {scenario}: {scenario_eval_hist[scenario]}")

    print("Accuracy by scenario:")
    for scenario in scenario_weights:
        s_total = by_scenario[scenario]["total"]
        s_correct = by_scenario[scenario]["correct"]
        if s_total > 0:
            print(f"  {scenario}: {s_correct}/{s_total} ({100 * s_correct / s_total:.1f}%)")

    print("Accuracy by carry count:")
    for carry_count in sorted(by_carry.keys()):
        c_total = by_carry[carry_count]["total"]
        c_correct = by_carry[carry_count]["correct"]
        if c_total > 0:
            print(f"  carry {carry_count}: {c_correct}/{c_total} ({100 * c_correct / c_total:.1f}%)")

    os.makedirs("results", exist_ok=True)
    with open("results/llm_output.txt", "w") as f:
        f.write(f"{NUM_DIGITS}-DIGIT EVAL REPORT\n")
        f.write(f"Total cases: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {100 * correct / total:.1f}%\n")
        f.write(f"Throughput: {total / elapsed:.1f} samples/sec\n\n")

        f.write("Scenario mix in evaluation:\n")
        for scenario in scenario_weights:
            f.write(f"  {scenario}: {scenario_eval_hist[scenario]}\n")

        f.write("\nAccuracy by scenario:\n")
        for scenario in scenario_weights:
            s_total = by_scenario[scenario]["total"]
            s_correct = by_scenario[scenario]["correct"]
            if s_total > 0:
                f.write(f"  {scenario}: {s_correct}/{s_total} ({100 * s_correct / s_total:.1f}%)\n")

        f.write("\nAccuracy by carry count:\n")
        for carry_count in sorted(by_carry.keys()):
            c_total = by_carry[carry_count]["total"]
            c_correct = by_carry[carry_count]["correct"]
            if c_total > 0:
                f.write(f"  carry {carry_count}: {c_correct}/{c_total} ({100 * c_correct / c_total:.1f}%)\n")

        f.write("\nSample errors (up to 10):\n")
        for scenario, a, b, pred_sum, actual_sum in errors[:10]:
            pred_str = pred_sum if pred_sum is not None else "None"
            f.write(
                f"  [{scenario}] {a:0{NUM_DIGITS}d}+{b:0{NUM_DIGITS}d}={pred_str} "
                f"should be {actual_sum:0{RESULT_DIGITS}d}\n"
            )


if __name__ == "__main__":
    main()
