#!/bin/bash
# Experiment: Fine-grained overhead measurement
# Measures CUDA + NVML + psutil query times per step/phase
 
SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
 
BATCH_MAX=128
BATCH_MID=64
BATCH_MIN=32
 
OUTPUT_DIR="${HOME}/COMP597-starter-code/results/basic_resources_logs_overhead_nw0_70400_v2"
mkdir -p ${OUTPUT_DIR}
 
for BS in $BATCH_MAX $BATCH_MID $BATCH_MIN; do
    echo "=== Running overhead measurement, batch_size=${BS} ==="
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
            --data_configs.fakeimagenet.folder "${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train" \
            --trainer_stats basic_resources_stats_overhead \
            --trainer_stats_configs.basic_resources.output_dir ${OUTPUT_DIR}/bs${BS}_run${RUN}
    done
done



