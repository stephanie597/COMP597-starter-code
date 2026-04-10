#!/bin/bash
# Experiment 1: End-to-end time baseline (NO instrumentation overhead)
# Run 3 times per batch size and average results manually.
# Replace BATCH_MAX, BATCH_MID, BATCH_MIN with your 3 chosen batch sizes (powers of 2).

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

BATCH_MAX=128   # <-- set your max power-of-2 batch size
BATCH_MID=64   # <-- half of max
BATCH_MIN=32   # <-- quarter of max

for BS in $BATCH_MAX $BATCH_MID $BATCH_MIN; do
    echo "=== Running end-to-end time baseline, batch_size=${BS} ==="
    for RUN in 1 2 3; do
        echo "--- Run ${RUN}/3 ---"
        ${SCRIPTS_DIR}/srun.sh \
            --logging.level INFO \
            --model resnet152 \
            --data fakeimagenet \
            --trainer resnet_simple \
            --batch_size ${BS} \
            --data_configs.dataset.load_num_proc 0 \
            --learning_rate 0.01 \
            --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
            --trainer_stats end_to_end_time_stats
    done
done

