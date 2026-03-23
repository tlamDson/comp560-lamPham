#!/bin/bash
set -e

CONFIG="config/basic.py"
LOG_DIR="/dev/shm/8_digits_training_logs"
NUM_RUNS=5
TRAIN_SCRIPT="../common/train.py"
VERIFY_SCRIPT="sample_and_verify_linux.py"

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

for i in $(seq 1 $NUM_RUNS)
do
    echo ">>> Trial $i / $NUM_RUNS — Training..."
    { time python -u "$TRAIN_SCRIPT" "$CONFIG" ; } > "$LOG_DIR/train_$i.log" 2>&1

    echo "    Training done. Running sample & verify..."
    python -u "$VERIFY_SCRIPT" > "$LOG_DIR/verify_$i.log" 2>&1

    echo "    Trial $i complete."
    echo ""
done

echo "============================================================"
echo "  All $NUM_RUNS trials finished at $(date)"
echo "============================================================"
echo ""

echo "----------------------------------------------------------"
printf "%-6s | %-14s | %-14s | %-10s\n" "Trial" "Real Time" "Final MFU (%)" "Accuracy (%)"
echo "----------------------------------------------------------"

for i in $(seq 1 $NUM_RUNS)
do
    REAL_TIME_RAW=$(grep "^real" "$LOG_DIR/train_$i.log" | awk '{print $2}')
    MFU=$(grep "mfu" "$LOG_DIR/train_$i.log" | tail -n 1 | grep -oP 'mfu\s+\K[0-9]+\.[0-9]+' || echo "N/A")
    ACC=$(grep "^Accuracy:" "$LOG_DIR/verify_$i.log" | grep -oP '[0-9]+\.[0-9]+' | head -1 || echo "N/A")

    printf "  %-4s | %-14s | %-14s | %-10s\n" "$i" "$REAL_TIME_RAW" "$MFU" "$ACC"
done

echo "----------------------------------------------------------"
echo ""

echo "--- Per-Carry Accuracy Summary (across all trials) ---"
for carry in $(seq 0 8)
do
    vals=()
    for i in $(seq 1 $NUM_RUNS)
    do
        v=$(grep "carry $carry:" "$LOG_DIR/verify_$i.log" | grep -oP '\(([0-9]+\.[0-9]+)%\)' | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        vals+=("$v")
    done
    printf "  carry %d : %s\n" "$carry" "${vals[*]}"
done

echo ""
echo "Logs saved in: $LOG_DIR"
echo "Done at: $(date)"
