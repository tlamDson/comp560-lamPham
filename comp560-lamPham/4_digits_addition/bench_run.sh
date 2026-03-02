#!/bin/bash
# ==============================================================================
# bench_run.sh — Automated 5-Run Training + Verification Benchmark
#
# Runs training 5 times, then runs sample_and_verify_linux.py after each,
# collecting: Real Time, Final MFU (%), and Accuracy (%).
# Outputs a summary table with Mean and Standard Deviation.
#
# Usage:
#   chmod +x bench_run.sh
#   ./bench_run.sh
# ==============================================================================

set -e

# -------------------------------- Configuration --------------------------------
CONFIG="config/basic.py"
LOG_DIR="/dev/shm/training_logs"
NUM_RUNS=5

TRAIN_SCRIPT="../../comp560-nanoGPT/train.py"
CONFIGURATOR="../../comp560-nanoGPT/configurator.py"
VERIFY_SCRIPT="sample_and_verify_linux.py"

# Ensure log directory exists and is clean
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  AUTOMATED BENCHMARK: $NUM_RUNS Training + Verify Runs"
echo "============================================================"
echo "Config       : $CONFIG"
echo "Log directory: $LOG_DIR"
echo "Start time   : $(date)"
echo "============================================================"
echo ""

# -------------------------------- Run Loop ------------------------------------
for i in $(seq 1 $NUM_RUNS)
do
    echo ">>> Trial $i / $NUM_RUNS — Training..."

    # Run training with 'time', capture everything (stdout+stderr) to log
    { time NANOGPT_CONFIG="$CONFIGURATOR" \
      python -u "$TRAIN_SCRIPT" "$CONFIG" ; } > "$LOG_DIR/train_$i.log" 2>&1

    echo "    Training done. Running sample & verify..."

    # Run verification, output to separate log
    python -u "$VERIFY_SCRIPT" > "$LOG_DIR/verify_$i.log" 2>&1

    echo "    Trial $i complete."
    echo ""
done

echo "============================================================"
echo "  All $NUM_RUNS trials finished at $(date)"
echo "============================================================"
echo ""

# -------------------------------- Extract Metrics -----------------------------
# Arrays to hold values for stats calculation
declare -a REAL_TIMES_SEC=()
declare -a MFUS=()
declare -a ACCURACIES=()
declare -a REAL_TIMES_DISPLAY=()

echo "----------------------------------------------------------"
printf "%-6s | %-14s | %-14s | %-10s\n" "Trial" "Real Time" "Final MFU (%)" "Accuracy (%)"
echo "----------------------------------------------------------"

for i in $(seq 1 $NUM_RUNS)
do
    # --- Extract Real Time ---
    # 'time' output format: "real    1m23.456s"
    REAL_TIME_RAW=$(grep "^real" "$LOG_DIR/train_$i.log" | awk '{print $2}')
    REAL_TIMES_DISPLAY+=("$REAL_TIME_RAW")

    # Convert to seconds for averaging: e.g., "1m23.456s" -> 83.456
    REAL_SEC=$(echo "$REAL_TIME_RAW" | sed 's/s$//' | awk -F'm' '{
        if (NF == 2) printf "%.3f", $1 * 60 + $2;
        else printf "%.3f", $1;
    }')
    REAL_TIMES_SEC+=("$REAL_SEC")

    # --- Extract Final MFU ---
    # train.py prints: "iter 9000: loss 0.0123, time 45.00ms, mfu 30.15%"
    MFU=$(grep "mfu" "$LOG_DIR/train_$i.log" | tail -n 1 | \
          grep -oP 'mfu\s+\K[0-9]+\.[0-9]+' || echo "N/A")
    MFUS+=("$MFU")

    # --- Extract Accuracy ---
    # sample_and_verify_linux.py prints: "Accuracy: 95.0%"
    ACC=$(grep "^Accuracy:" "$LOG_DIR/verify_$i.log" | \
          grep -oP '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
    ACCURACIES+=("$ACC")

    printf "  %-4s | %-14s | %-14s | %-10s\n" "$i" "$REAL_TIME_RAW" "$MFU" "$ACC"
done

echo "----------------------------------------------------------"
echo ""

# -------------------------------- Statistics ----------------------------------
# Use awk to calculate mean and standard deviation

calc_stats() {
    # Takes values as arguments, prints "mean stddev"
    local vals=("$@")
    echo "${vals[@]}" | tr ' ' '\n' | awk '
    {
        sum += $1
        sumsq += $1 * $1
        n++
    }
    END {
        if (n > 0) {
            mean = sum / n
            if (n > 1)
                stddev = sqrt((sumsq - n * mean * mean) / (n - 1))
            else
                stddev = 0
            printf "%.3f %.3f\n", mean, stddev
        }
    }'
}

# Calculate stats for each metric
TIME_STATS=$(calc_stats "${REAL_TIMES_SEC[@]}")
TIME_MEAN=$(echo "$TIME_STATS" | awk '{print $1}')
TIME_STD=$(echo "$TIME_STATS" | awk '{print $2}')

MFU_STATS=$(calc_stats "${MFUS[@]}")
MFU_MEAN=$(echo "$MFU_STATS" | awk '{print $1}')
MFU_STD=$(echo "$MFU_STATS" | awk '{print $2}')

ACC_STATS=$(calc_stats "${ACCURACIES[@]}")
ACC_MEAN=$(echo "$ACC_STATS" | awk '{print $1}')
ACC_STD=$(echo "$ACC_STATS" | awk '{print $2}')

# Convert mean time back to min:sec for display
TIME_MEAN_MIN=$(echo "$TIME_MEAN" | awk '{
    m = int($1 / 60)
    s = $1 - m * 60
    printf "%dm%.3fs", m, s
}')

echo "=========================================================="
echo "  STATISTICAL SUMMARY ($NUM_RUNS runs)"
echo "=========================================================="
printf "  %-20s : %s (%.3fs ± %.3fs)\n" "Real Time (mean±σ)" "$TIME_MEAN_MIN" "$TIME_MEAN" "$TIME_STD"
printf "  %-20s : %.2f%% ± %.2f%%\n" "MFU (mean±σ)" "$MFU_MEAN" "$MFU_STD"
printf "  %-20s : %.1f%% ± %.1f%%\n" "Accuracy (mean±σ)" "$ACC_MEAN" "$ACC_STD"
echo "=========================================================="
echo ""

# -------------------------------- Per-Carry Accuracy Summary ------------------
echo "--- Per-Carry Accuracy Summary (across all trials) ---"

for carry in 0 1 2 3 4
do
    CARRY_ACCS=()
    for i in $(seq 1 $NUM_RUNS)
    do
        CACC=$(grep "carry $carry:" "$LOG_DIR/verify_$i.log" | \
               grep -oP '\(([0-9]+\.[0-9]+)%\)' | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        CARRY_ACCS+=("$CACC")
    done

    CARRY_STATS=$(calc_stats "${CARRY_ACCS[@]}")
    CARRY_MEAN=$(echo "$CARRY_STATS" | awk '{print $1}')
    CARRY_STD=$(echo "$CARRY_STATS" | awk '{print $2}')
    printf "  carry %d : %.1f%% ± %.1f%%\n" "$carry" "$CARRY_MEAN" "$CARRY_STD"
done

echo ""
echo "Logs saved in: $LOG_DIR"
echo "Done at: $(date)"
