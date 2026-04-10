#!/bin/bash
# Experiment 3: Fine-grained monitoring (GPU util, CPU util, GPU memory timelines + per-phase timing)

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

BATCH_MAX=128
BATCH_MID=64
BATCH_MIN=32

# Store results in home directory so they're accessible from login node
OUTPUT_DIR="${HOME}/COMP597-starter-code/results/basic_resources_logs_nw1_70400_v2"
mkdir -p ${OUTPUT_DIR}

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
            --data_configs.dataset.load_num_proc 1 \
            --learning_rate 0.01 \
            --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
            --trainer_stats basic_resources_stats \
            --trainer_stats_configs.basic_resources.output_dir ${OUTPUT_DIR}/bs${BS}_run${RUN}
    done
done




