#!/bin/bash
# Experiment 2: End-to-end energy baseline (single CodeCarbon measurement)
 
SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
 
BATCH_MAX=128
BATCH_MID=64
BATCH_MIN=32
 
for BS in $BATCH_MAX $BATCH_MID $BATCH_MIN; do
    OUTPUT_DIR="${HOME}/COMP597-starter-code/results/energy_baseline_logs/bs${BS}"
    mkdir -p ${OUTPUT_DIR}
    echo "=== Running end-to-end energy baseline, batch_size=${BS} ==="
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
            --trainer_stats end_to_end_energy_stats \
            --trainer_stats_configs.codecarbon.run_num ${RUN} \
            --trainer_stats_configs.codecarbon.project_name resnet152_energy_bs${BS} \
            --trainer_stats_configs.codecarbon.output_dir ${OUTPUT_DIR}
    done
done