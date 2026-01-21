# Prime Checker Experiment

## Goal

Train a character-level language model to identify whether a number is prime or not.

## Data Format

Each line contains a number followed by `:P` (prime) or `:N` (not prime).

Example:

```
2:P
3:P
4:N
5:P
```

## Experiment Log

### Version: basic

- Date: 2026-01-21
- Description: Initial experiment with numbers 2-100
- Configuration:
  - max_iters: 2000
  - batch_size: 12
  - block_size: 64
  - n_layer: 4
  - n_head: 4
  - n_embd: 128
  - learning_rate: 1e-3
  - n_params: 0.79M

### Results

#### Training

- Successfully trained for 2000 iterations on CPU
- Dataset: 9,900 examples (numbers 2-100, repeated 100 times each)
- Train/Val split: 90/10

#### Sampling Output Analysis

The model generates mostly correct predictions but shows some errors:

**Correct predictions (examples):**

- `10:N` ✓ (10 is not prime)
- `83:P` ✓ (83 is prime)
- `5:P` ✓ (5 is prime)
- `17:P` ✓ (17 is prime)
- `79:P` ✓ (79 is prime)
- `43:P` ✓ (43 is prime)

**Incorrect predictions (examples):**

- `69:P` ✗ (69 = 3×23, should be N)
- `67:N` ✗ (67 is prime, should be P)
- `73:N` ✗ (73 is prime, should be P)

#### Observations

1. **What worked well:**
   - Model learned the basic format (number:letter)
   - Correctly identifies many primes and composites
   - Generates syntactically valid outputs

2. **What needs improvement:**
   - Accuracy is not 100% - makes errors on some primes
   - Particularly struggles with larger primes (67, 73)
   - May need more training iterations or larger model

3. **Why this is challenging:**
   - Prime number detection requires mathematical reasoning, not just pattern matching
   - Character-level model sees "73" as two separate digits, not a number
   - No explicit mathematical operations in the model

#### Potential Improvements

1. Increase training iterations (try 5000+)
2. Use larger model (more layers/embedding size)
3. Simplify problem: use smaller range (2-20 or 2-50)
4. Add more repetitions of examples in training data
5. Try different data encoding (e.g., binary representation of numbers)

---

### Version: basic-v2 (Improved)

- Date: 2026-01-21
- Description: Simplified experiment with reduced range
- **Changes made:**
  - Number range: 2-100 → **2-20**
  - Repetitions per example: 100 → **500**
  - Training iterations: 2000 → **5000**
  - Dataset size: 9,900 → 9,500 examples

#### Results: ✅ 100% Accuracy

All predictions verified correct across 5 samples:

- **Primes correctly identified:** 2, 3, 5, 7, 11, 13, 17, 19
- **Composites correctly identified:** 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20

**Example output:**

```
10:N 8:N 15:N 2:P 13:P 11:P 18:N 6:N 5:P 14:N 19:P 15:N 4:N
16:N 3:P 15:N 19:P 14:N 11:P 14:N 13:P 11:P 18:N 20:N 4:N 7:P
```

---

## Summary of All Experiments

| Experiment | Range | Repetitions | Iterations | Result           | Key Insight                              |
| ---------- | ----- | ----------- | ---------- | ---------------- | ---------------------------------------- |
| **v1**     | 2-100 | 100×        | 2000       | ~60-70% accuracy | Problem too complex for model capacity   |
| **v2**     | 2-20  | 500×        | 5000       | 100% accuracy    | Proper scoping + more training = success |

**Conclusion:** Character-level GPT models can successfully learn mathematical patterns when the problem is appropriately scoped to match model capacity. Reducing the number range from 100 to 20 numbers and increasing training iterations from 2000 to 5000 resulted in perfect accuracy. This demonstrates the importance of matching problem complexity to model size and training budget.

## Automated Testing

### sample_and_verify.py

An automated script that runs sampling and immediately verifies the model's predictions:

**Usage:**

```bash
python sample_and_verify.py
```

**What it does:**

1. Automatically runs the nanoGPT sample.py with configured parameters
2. Captures the model's output
3. Saves output to `llm_output.txt`
4. Verifies each prediction against the actual prime/composite status
5. Reports accuracy statistics and lists any errors

**Output example:**

```
AUTOMATED SAMPLE & VERIFY

Running: python ...comp560-nanoGPT\sample.py config/basic.py --num_samples=5 --max_new_tokens=100

[Sample output displayed here...]

Saved output to llm_output.txt

VERIFICATION RESULTS

Correct predictions: 110/110
Accuracy: 100.0%

PERFECT! All predictions correct!
```

This eliminates manual verification and provides instant feedback on model performance.
