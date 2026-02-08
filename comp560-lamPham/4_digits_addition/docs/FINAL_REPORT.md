# 4-Digit Addition Experiment - Final Report

## Overview

This experiment extends the 3-digit addition task to 4-digit numbers, testing the model's ability to learn more complex arithmetic.

---

## Task Description

**Input format**: `AAAA+BBBB=`  
**Output format**: `CCCCC` (5-digit result, zero-padded)

**Example**: `1234+5678=06912`

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| n_layer | 4 |
| n_head | 4 |
| n_embd | 256 |
| block_size | 20 |
| batch_size | 256 |
| max_iters | 8000 |
| learning_rate | 4e-3 |

---

## Dataset

- **Training examples**: ~225,000 (90% of total)
- **Validation examples**: ~25,000 (10% of total)
- **Balanced by carry count**: 50,000 examples per carry class (0-4 carries)

---

## Results

_(To be filled after experiments)_

### Training Metrics

- **Final training loss**: TBD
- **Final validation loss**: TBD
- **Training time**: TBD

### Accuracy by Carry Count

| Carries | Accuracy |
|---------|----------|
| 0 | TBD |
| 1 | TBD |
| 2 | TBD |
| 3 | TBD |
| 4 | TBD |

### Overall Accuracy

- **Total correct**: TBD
- **Accuracy**: TBD%

---

## Analysis

_(To be filled after experiments)_

---

## Comparison with 3-Digit Addition

| Metric | 3-digit | 4-digit |
|--------|---------|---------|
| Training time | ~3 min | TBD |
| Final accuracy | 100% | TBD |
| Dataset size | 240k | 250k |

---

## Conclusions

_(To be filled after experiments)_
