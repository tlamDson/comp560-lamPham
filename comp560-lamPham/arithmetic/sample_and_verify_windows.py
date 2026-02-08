"""
Sample and automatically verify 3-digit addition results.
This script prompts the model and verifies the predicted sum.

Windows version - uses CPU for sampling (GPU sampling can be unstable on Windows).
"""
import os
import random
import re
import subprocess
import sys

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

def format_prompt(a, b):
    return f"{a:03d}+{b:03d}="

def extract_prediction(output, prompt):
    idx = output.find(prompt)
    if idx == -1:
        return None
    after = output[idx + len(prompt):]
    match = re.search(r"(\d{4})", after)
    if not match:
        return None
    return match.group(1)

def verify_predictions(cases, predict_fn):
    correct = 0
    total = 0
    errors = []

    by_carry = {
        0: {"correct": 0, "total": 0},
        1: {"correct": 0, "total": 0},
        2: {"correct": 0, "total": 0},
        3: {"correct": 0, "total": 0},
    }

    for a, b in cases:
        predicted_sum_str = predict_fn(a, b)
        actual_sum = a + b
        carries = count_carries(a, b)
        by_carry[carries]["total"] += 1
        total += 1

        if predicted_sum_str is None:
            errors.append((a, b, None, actual_sum))
            continue

        try:
            predicted_sum = int(predicted_sum_str)
        except ValueError:
            errors.append((a, b, predicted_sum_str, actual_sum))
            continue

        if predicted_sum == actual_sum:
            correct += 1
            by_carry[carries]["correct"] += 1
        else:
            errors.append((a, b, predicted_sum_str, actual_sum))

    return correct, total, errors, by_carry

# Configuration - Windows paths (using raw strings with backslashes)
NANOGPT_PATH = r"..\..\comp560-nanoGPT"
CONFIG_FILE = r"config\basic.py"
MAX_NEW_TOKENS = 4
EVAL_PER_CARRY = 20
SEED = 42
TEMPERATURE = 0.8  # Slightly increase for stability
TOP_K = 200  # Add top_k sampling

print("AUTOMATED SAMPLE & VERIFY (ADDITION) - Windows")

sample_script = os.path.join(NANOGPT_PATH, "sample.py")
base_cmd = [
    sys.executable,
    "-u",
    sample_script,
    CONFIG_FILE,
    "--num_samples=1",
    f"--max_new_tokens={MAX_NEW_TOKENS}",
]

env = os.environ.copy()
configurator_path = os.path.abspath(os.path.join(NANOGPT_PATH, "configurator.py"))
env["NANOGPT_CONFIG"] = configurator_path

try:
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

    def predict_fn(a, b):
        prompt = format_prompt(a, b)
        cmd = base_cmd + [
            f"--start={prompt}",
            f"--temperature={TEMPERATURE}",
            f"--top_k={TOP_K}",
            "--device=cpu"  # Use CPU on Windows (GPU sampling can be unstable)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, timeout=60)
            return extract_prediction(result.stdout, prompt)
        except subprocess.TimeoutExpired:
            print(f"  Timeout on {prompt}")
            return None
        except subprocess.CalledProcessError as e:
            print(f"  Error on {prompt}: {e.returncode}")
            if e.stderr:
                print(f"    {e.stderr[:200]}")
            return None

    print("\nRunning prompt-based evaluation...")
    print(f"Evaluation cases: {len(cases)} (carry distribution: {counts})\n")

    correct, total, errors, by_carry = verify_predictions(cases, predict_fn)

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

    with open(r"results\llm_output.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")

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

    print("\nSaved report to results\\llm_output.txt")

except subprocess.CalledProcessError as e:
    print(f"Error running sample.py: {e}")
    if e.stderr:
        print(e.stderr)
except Exception as e:
    print(f"Error: {e}")
