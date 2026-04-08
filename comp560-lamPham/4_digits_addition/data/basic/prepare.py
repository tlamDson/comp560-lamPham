"""
Prepare a 4-digit addition dataset using a data-centric mix:
- 70% stratified random lengths
- 10% cascading carry cases
- 10% extreme length imbalance
- 10% boundary/zero-heavy cases

Examples use fixed-width operands and reversed targets to remain
compatible with the existing training and verifier pipeline.
"""

import os
import pickle
import random
from collections import Counter

import numpy as np

SEED = 42
NUM_DIGITS = 4
RESULT_DIGITS = NUM_DIGITS + 1
TOTAL_EXAMPLES = 250000
TRAIN_SPLIT = 0.9

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


def format_example(a, b):
    sum_str = f"{a + b:0{RESULT_DIGITS}d}"
    reversed_sum = sum_str[::-1]
    return f"{a:0{NUM_DIGITS}d}+{b:0{NUM_DIGITS}d}={reversed_sum}\n"


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


def scenario_counts(total_examples):
    names = list(SCENARIO_WEIGHTS.keys())
    counts = {name: int(total_examples * SCENARIO_WEIGHTS[name]) for name in names}
    assigned = sum(counts.values())
    remainder = total_examples - assigned
    for name in sorted(names, key=lambda n: SCENARIO_WEIGHTS[n], reverse=True)[:remainder]:
        counts[name] += 1
    return counts


def canonical_pair(a, b):
    return (a, b) if a <= b else (b, a)


def build_examples(total_examples):
    generators = {
        "stratified_random": stratified_random_pair,
        "cascading_carries": cascading_carry_pair,
        "extreme_imbalance": extreme_imbalance_pair,
        "boundary_zeros": boundary_zeros_pair,
    }

    counts = scenario_counts(total_examples)
    examples = []
    seen = set()

    carry_hist = Counter()
    len_a_hist = Counter()
    len_b_hist = Counter()
    scenario_hist = Counter()

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
            examples.append((a, b, scenario))
            carry_hist[count_carries(a, b)] += 1
            len_a_hist[len(str(a))] += 1
            len_b_hist[len(str(b))] += 1
            scenario_hist[scenario] += 1
            accepted += 1

    random.shuffle(examples)
    return examples, carry_hist, len_a_hist, len_b_hist, scenario_hist


random.seed(SEED)
examples, carry_hist, len_a_hist, len_b_hist, scenario_hist = build_examples(TOTAL_EXAMPLES)
data_lines = [format_example(a, b) for a, b, _ in examples]
data = "".join(data_lines)

print(f"length of dataset in characters: {len(data):,}")
print(f"number of examples: {len(data_lines):,}")

print("scenario distribution:")
for name in SCENARIO_WEIGHTS:
    print(f"  {name}: {scenario_hist[name]}")

print("carry distribution (observed):")
for c in range(NUM_DIGITS + 1):
    if carry_hist[c] > 0:
        print(f"  carry {c}: {carry_hist[c]}")

print("length distribution of operand A (digits):")
for l in range(1, NUM_DIGITS + 1):
    if len_a_hist[l] > 0:
        print(f"  len {l}: {len_a_hist[l]}")

print("length distribution of operand B (digits):")
for l in range(1, NUM_DIGITS + 1):
    if len_b_hist[l] > 0:
        print(f"  len {l}: {len_b_hist[l]}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


n_examples = len(data_lines)
split_idx = int(n_examples * TRAIN_SPLIT)
train_lines = data_lines[:split_idx]
val_lines = data_lines[split_idx:]

if set(train_lines) & set(val_lines):
    raise RuntimeError("Leakage detected: train and val share duplicate examples")

train_data = "".join(train_lines)
val_data = "".join(val_lines)

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

base = os.path.dirname(__file__)
train_ids.tofile(os.path.join(base, "train.bin"))
val_ids.tofile(os.path.join(base, "val.bin"))

meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
    "num_digits": NUM_DIGITS,
    "result_digits": RESULT_DIGITS,
    "scenario_weights": SCENARIO_WEIGHTS,
}
with open(os.path.join(base, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nFirst few examples:")
print("".join(data_lines[:3]))
