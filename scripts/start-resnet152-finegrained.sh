#!/bin/bash
# Experiment 3: Fine-grained monitoring (GPU util, CPU util, GPU memory timelines + per-phase timing)
# Replace BATCH_MAX, BATCH_MID, BATCH_MIN with your 3 chosen batch sizes (powers of 2).

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

BATCH_MAX=64   # <-- set your max power-of-2 batch size
BATCH_MID=32   # <-- half of max
BATCH_MIN=16   # <-- quarter of max

OUTPUT_DIR='${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs'

for BS in $BATCH_MAX $BATCH_MID $BATCH_MIN; do
    echo "=== Running fine-grained monitoring, batch_size=${BS} ==="
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
            --trainer_stats basic_resources_stats \
            --trainer_stats_configs.basic_resources.output_dir ${OUTPUT_DIR}
    done
done


