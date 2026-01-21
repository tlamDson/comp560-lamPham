"""
Sample and automatically verify the results.
This script runs sampling and immediately verifies the output.
"""

import subprocess
import sys
import os

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def verify_predictions(text):
    predictions = text.strip().split()
    correct = 0
    total = 0
    errors = []
    
    for pred in predictions:
        if ':' not in pred or pred == '---------------':
            continue
        try:
            num_str, label = pred.split(':')
            num = int(num_str)
            correct_label = 'P' if is_prime(num) else 'N'
            
            if label == correct_label:
                correct += 1
            else:
                errors.append((num, label, correct_label))
            total += 1
        except:
            continue
    
    return correct, total, errors

# Configuration
NANOGPT_PATH = r"..\..\comp560-nanoGPT"
CONFIG_FILE = r"config/basic.py"
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 100

print("AUTOMATED SAMPLE & VERIFY")

# Build the command
sample_script = os.path.join(NANOGPT_PATH, "sample.py")
cmd = [
    sys.executable,
    "-u",
    sample_script,
    CONFIG_FILE,
    f"--num_samples={NUM_SAMPLES}",
    f"--max_new_tokens={MAX_NEW_TOKENS}"
]

print(f"\nRunning: {' '.join(cmd)}\n")

# Set environment variable for nanoGPT configurator
env = os.environ.copy()
configurator_path = os.path.abspath(os.path.join(NANOGPT_PATH, "configurator.py"))
env['NANOGPT_CONFIG'] = configurator_path

# Run sampling and capture output
try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    output = result.stdout
    
    # Print the sample output
    print(output)
    
    # Save to file
    with open('llm_output.txt', 'w') as f:
        f.write(output)
    print(f"\nSaved output to llm_output.txt\n")
    
    # Verify predictions
    print("\nVERIFICATION RESULTS")
    
    correct, total, errors = verify_predictions(output)
    
    if total == 0:
        print("No predictions found in output")
    else:
        accuracy = (correct / total) * 100
        
        print(f"\nCorrect predictions: {correct}/{total}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        if errors:
            print(f"\nErrors found: {len(errors)}")
            for num, pred, correct_label in errors:
                status = "should be PRIME" if correct_label == 'P' else "should be COMPOSITE"
                print(f"  {num}:{pred} should be {num}:{correct_label} ({status})")
        else:
            print("\nPERFECT! All predictions correct!")

except subprocess.CalledProcessError as e:
    print(f"Error running sample.py: {e}")
    if e.stderr:
        print(e.stderr)
except Exception as e:
    print(f"Error: {e}")
