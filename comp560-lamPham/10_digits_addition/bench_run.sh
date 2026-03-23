#!/bin/bash
set -e

CONFIG="config/basic.py"
LOG_DIR="/dev/shm/10_digits_training_logs"
NUM_RUNS=5
TRAIN_SCRIPT="../common/train.py"
VERIFY_SCRIPT="sample_and_verify_linux.py"

rm -rf ""
mkdir -p ""

echo "============================================================"
echo "  AUTOMATED BENCHMARK:  Training + Verify Runs"
echo "============================================================"
echo "Config       : "
echo "Log directory: "
echo "Start time   : Mon Mar 23 13:06:31 EDT 2026"
echo "============================================================"
echo ""

for i in 1
do
    echo ">>> Trial  /  — Training..."
    { time python -u "" "" ; } > "/train_.log" 2>&1

    echo "    Training done. Running sample & verify..."
    python -u "" > "/verify_.log" 2>&1

    echo "    Trial  complete."
    echo ""
done

echo "============================================================"
echo "  All  trials finished at Mon Mar 23 13:06:31 EDT 2026"
echo "============================================================"
echo ""

echo "----------------------------------------------------------"
printf "%-6s | %-14s | %-14s | %-10s\n" "Trial" "Real Time" "Final MFU (%)" "Accuracy (%)"
echo "----------------------------------------------------------"

for i in 1
do
    REAL_TIME_RAW=
    MFU=N/A
    ACC=

    printf "  %-4s | %-14s | %-14s | %-10s\n" "" "" "" ""
done

echo "----------------------------------------------------------"
echo ""

echo "--- Per-Carry Accuracy Summary (across all trials) ---"
for carry in 0
1
2
3
4
5
6
7
8
9
10
do
    vals=()
    for i in 1
    do
        v=N/A
        vals+=("")
    done
    printf "  carry %d : %s\n" "" ""
done

echo ""
echo "Logs saved in: "
echo "Done at: Mon Mar 23 13:06:31 EDT 2026"
