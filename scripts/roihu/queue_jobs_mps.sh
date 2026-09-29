#!/bin/bash -l

ML=3
gpus=4
total_cpus=288  # whole node's cores: 4 GH200 x 72-core Grace CPU
# run with TALP enabled
# TALP=1 ./queue_jobs_mps.sh

for NODES in 1; do
    for N in 4 8 16 32; do
        cpus=$((total_cpus / N))
        sbatch --job-name="run_Elmer_roihu_N${NODES}_n${N}_c${cpus}_ML${ML}_MPS" \
               --nodes=${NODES} \
               --ntasks-per-node=${N} \
               --cpus-per-task=${cpus} \
               --time="00:60:00" \
               --output="logs/%x_%j.out" \
               --error="logs/%x_%j.err" \
               --mem=0 \
               --gres=gpu:gh200:${gpus} \
               --partition=gpumedium \
               --account=project_2001659 \
               run_mps.sh ${ML}
    done
done

#            --exclusive \
