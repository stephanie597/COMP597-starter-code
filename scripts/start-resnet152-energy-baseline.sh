#!/bin/bash
# Experiment 2: End-to-end energy baseline (single CodeCarbon measurement, minimal overhead)
# Replace BATCH_MAX, BATCH_MID, BATCH_MIN with your 3 chosen batch sizes (powers of 2).

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

BATCH_MAX=64   # <-- set your max power-of-2 batch size
BATCH_MID=32   # <-- half of max
BATCH_MIN=16   # <-- quarter of max

OUTPUT_DIR='${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/energy_baseline_logs'

for BS in $BATCH_MAX $BATCH_MID $BATCH_MIN; do
    echo "=== Running end-to-end energy baseline, batch_size=${BS} ==="
    for RUN in 1 2 3; do
        echo "--- Run ${RUN}/3 ---"
        ${SCRIPTS_DIR}/srun.sh \
            --logging.level INFO \
            --model resnet152 \
            --data fakeimagenet \
            --trainer resnet_simple \
            --batch_size ${BS} \
            --learning_rate 1e-6 \
            --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
            --trainer_stats end_to_end_energy_stats \
            --trainer_stats_configs.codecarbon.run_num ${RUN} \
            --trainer_stats_configs.codecarbon.project_name resnet152_energy_bs${BS} \
            --trainer_stats_configs.codecarbon.output_dir ${OUTPUT_DIR}
    done
done

