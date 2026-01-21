# Experiment Report: Character-Level Prime Number Classification

**Author:** Lam Pham  
**Date:** January 21, 2026  
**Course:** COMP 560

## 1. Introduction

This report documents an experiment in training a small GPT-based language model to identify prime numbers using only character-level information. The goal was to explore whether a transformer model could learn mathematical properties through pattern recognition alone, without explicit arithmetic operations.

## 2. Experimental Design

### 2.1 Data Stream Choice

I chose to train a model on **prime number classification** as my data stream. Each training example consists of a number followed by its classification:

- Format: `{number}:P` for prime numbers, `{number}:N` for non-prime numbers
- Example stream:
  ```
  2:P
  3:P
  4:N
  5:P
  6:N
  7:P
  8:N
  ```

This task was selected because:

1. It has a clear, deterministic answer for each input
2. It requires the model to learn number properties rather than just memorizing sequences
3. It's simple enough to evaluate success/failure visually
4. It provides an interesting test of whether transformers can learn mathematical concepts

### 2.2 Data Preparation

The `prepare.py` script generates training data as follows:

- **Range:** Numbers from 2 to 100
- **Method:** Programmatic generation using a prime-checking function
- **Dataset size:** 9,900 examples (each number repeated 100 times)
- **Split:** 90% training, 10% validation
- **Encoding:** Character-level (each digit and symbol mapped to an integer)
- **Output files:** `train.bin`, `val.bin`, `meta.pkl`

### 2.3 Model Configuration

Following the nanoGPT architecture for small experiments:

- **Model type:** GPT (decoder-only transformer)
- **Parameters:** 0.79M
- **Architecture:**
  - Layers: 4
  - Attention heads: 4
  - Embedding dimension: 128
  - Block size (context): 64 tokens
  - Dropout: 0.0
- **Training:**
  - Batch size: 12
  - Learning rate: 1e-3 → 1e-4 (decayed)
  - Iterations: 2000
  - Device: CPU
  - Time: ~5 minutes

## 3. Results

### 3.1 Training Process

The model trained successfully for 2000 iterations. The training process:

1. First run: 200 iterations (quick test) - verified workflow
2. Second run: 2000 iterations (full training) - actual experiment

### 3.2 Sample Outputs

Five samples were generated with max_new_tokens=100. Representative examples:

**Sample 1:**

```
10:N  83:P  51:N  24:N  56:N  79:P  50:N  8:N  54:N  5:P
17:P  50:N  99:N  69:P  82:N  93:N  7:P  67:N  5:P  43:P
73:N
```

**Sample 2:**

```
45:N  45:N  2:P  32:N  77:P  56:N  84:N  87:N  86:N  19:P
35:N  69:P  89:N  59:N  80:N  99:N  46:N  41:P  59:P  34:N
```

### 3.3 Accuracy Analysis

Manual verification of predictions:

| Number | Predicted | Actual | Correct? | Notes       |
| ------ | --------- | ------ | -------- | ----------- |
| 10     | N         | N      | ✓        |             |
| 83     | P         | P      | ✓        | Large prime |
| 5      | P         | P      | ✓        | Small prime |
| 17     | P         | P      | ✓        |             |
| 79     | P         | P      | ✓        | Large prime |
| 43     | P         | P      | ✓        |             |
| 69     | P         | N      | ✗        | 69 = 3×23   |
| 67     | N         | P      | ✗        | 67 is prime |
| 73     | N         | P      | ✗        | 73 is prime |
| 77     | P         | N      | ✗        | 77 = 7×11   |
| 89     | N         | P      | ✗        | 89 is prime |
| 59     | N         | P      | ✗        | 59 is prime |

**Estimated accuracy:** ~60-70% (based on sample inspection)

## 4. Analysis

### 4.1 What Worked Well

1. **Format learning:** The model perfectly learned the `{number}:{letter}` format
2. **Basic patterns:** Successfully identifies many obvious composites (even numbers, multiples of 5)
3. **Some primes:** Correctly identifies common small primes (2, 3, 5, 7, 17, 43)
4. **Syntactic validity:** Never produces malformed outputs

### 4.2 What Didn't Work

1. **Inconsistent accuracy:** Makes errors on both primes and composites
2. **Larger primes:** Struggles with primes above 50 (67, 73, 89)
3. **No perfect learning:** Even with 2000 iterations, doesn't achieve 100% accuracy

### 4.3 Why This Is Difficult

The challenge stems from fundamental limitations:

1. **Character-level encoding:** The model sees "73" as two separate characters '7' and '3', not as the number seventy-three
2. **No arithmetic:** The transformer has no built-in mathematical operations
3. **Pattern vs. logic:** Prime detection requires divisibility testing, not just pattern matching
4. **Limited context:** Block size of 64 limits how much previous data the model can reference

## 5. Next Steps

Based on these results, I identified two possible paths:

### Option A: Improve Current Experiment

- Increase iterations to 5000+
- Increase model size (more layers/embedding)
- Add more training data repetitions
- Try different learning rate schedules

### Option B: Simplify Problem (Recommended)

Create a simpler version to achieve better results:

- **Reduced range:** Numbers 2-20 only (fewer primes to learn)
- **More repetitions:** Repeat each example 500 times instead of 100
- **Longer training:** 5000 iterations

## 6. Conclusion

This experiment successfully demonstrated that a small GPT model can learn _some_ mathematical patterns through character-level training, but cannot achieve perfect accuracy on prime number classification with this configuration. The model learned the format and basic patterns (like identifying even numbers as composite) but struggled with true prime detection for larger numbers.

The experiment achieved its educational goals:

1. Set up and ran the nanoGPT training pipeline
2. Created a custom dataset and configuration
3. Observed the limitations of pattern-based learning for mathematical tasks
4. Identified areas for improvement

The next iteration will focus on a simpler problem (smaller number range) to demonstrate that the approach can work with appropriate problem scoping.

---

## 7. Appendix: Complete Sample Output

```
Sample 1:
10:N 83:P 51:N 24:N 56:N 79:P 50:N 8:N 54:N 5:P 17:P 50:N 99:N 69:P 82:N 93:N 7:P 67:N 5:P 43:P 73:N

Sample 2:
45:N 45:N 2:P 32:N 77:P 56:N 84:N 87:N 86:N 19:P 35:N 69:P 89:N 59:N 80:N 99:N 46:N 41:P 59:P 34:N 6

Sample 3:
30:N 76:N 97:N 70:N 2:P 80:N 57:N 51:N 31:P 33:N 19:P 33:N 89:P 32:N 89:P 83:P 98:N 92:N 22:N 10:N 3

Sample 4:
16:N 32:N 59:N 96:N 47:N 14:N 40:N 35:N 100:N 62:N 63:P 52:N 23:P 97:P 36:N 30:N 49:P 18:N 85:N 17:P

Sample 5:
14:N 36:N 13:P 70:N 59:N 23:P 74:N 6:N 80:N 83:P 51:N 70:N 60:N 64:N 67:N 42:N 32:N 46:N 95:N 67:P 9
```
