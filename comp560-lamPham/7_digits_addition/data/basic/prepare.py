"""
Prepare a 7-digit addition dataset for character-level modeling.
Examples use reversed targets to match current verifier/training setup.
"""
import os
import pickle
import random
import numpy as np

SEED = 42
NUM_DIGITS = 7
RESULT_DIGITS = NUM_DIGITS + 1
MIN_NUM = 1000000
MAX_NUM = 9999999
TARGET_PER_CARRY = 30000

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

random.seed(SEED)

target_by_carry = {c: TARGET_PER_CARRY for c in range(NUM_DIGITS + 1)}
counts = {c: 0 for c in range(NUM_DIGITS + 1)}
data_lines = []

while any(counts[c] < target_by_carry[c] for c in counts):
    a = random.randint(MIN_NUM, MAX_NUM)
    b = random.randint(MIN_NUM, MAX_NUM)
    carries = count_carries(a, b)
    if counts[carries] < target_by_carry[carries]:
        data_lines.append(format_example(a, b))
        counts[carries] += 1

random.shuffle(data_lines)

data = "".join(data_lines)
print(f"length of dataset in characters: {len(data):,}")
print(f"number of examples: {len(data_lines):,}")
print(f"carry distribution: {counts}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

base = os.path.dirname(__file__)
train_ids.tofile(os.path.join(base, "train.bin"))
val_ids.tofile(os.path.join(base, "val.bin"))

meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
with open(os.path.join(base, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nFirst few examples:")
print("".join(data_lines[:5]))
