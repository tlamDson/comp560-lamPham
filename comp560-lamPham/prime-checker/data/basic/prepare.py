"""
Prepare the prime number dataset for character-level language modeling.
Generates examples like: "2:P", "3:P", "4:N", "5:P", etc.
"""
import os
import pickle
import random
import numpy as np

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Generate data: numbers from 2 to 20 with their prime status (simplified range)
data_lines = []
for num in range(2, 21):
    label = "P" if is_prime(num) else "N"
    data_lines.append(f"{num}:{label}")

# Shuffle and repeat to create a larger dataset
random.seed(42)
# Repeat each example multiple times to create a larger training set
repeated_data = data_lines * 500  # More repetitions for better learning
random.shuffle(repeated_data)

# Join with newlines
data = '\n'.join(repeated_data) + '\n'

print(f"length of dataset in characters: {len(data):,}")
print(f"number of examples: {len(repeated_data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\nFirst few examples:")
print('\n'.join(repeated_data[:10]))
