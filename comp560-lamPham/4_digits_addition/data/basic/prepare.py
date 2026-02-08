"""
Prepare a 4-digit addition dataset for character-level modeling.
Examples: "1234+5678=06912"
"""
import os
import pickle
import random
import numpy as np

SEED = 42
MIN_NUM = 1000
MAX_NUM = 9999
TARGET_PER_CARRY = 50000  # Increased dataset size for more complex task

def count_carries(a, b):
    """Count the number of carries when adding two 4-digit numbers."""
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

def format_example(a, b):
    """Format as 4-digit + 4-digit = 5-digit (max sum is 19998)."""
    return f"{a:04d}+{b:04d}={a + b:05d}\n"

random.seed(SEED)

# Balance by carry count (0-4 possible carries for 4-digit addition)
target_by_carry = {0: TARGET_PER_CARRY, 1: TARGET_PER_CARRY, 2: TARGET_PER_CARRY, 3: TARGET_PER_CARRY, 4: TARGET_PER_CARRY}
counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
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

def decode(l):
    return "".join([itos[i] for i in l])

n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nFirst few examples:")
print("".join(data_lines[:5]))
