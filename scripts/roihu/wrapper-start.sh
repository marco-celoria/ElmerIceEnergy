#!/bin/bash

export CUDA_MPS_LOG_DIRECTORY=./pipe_${SLURM_JOB_ID}_${SLURM_NODEID}
export CUDA_MPS_PIPE_DIRECTORY=./pipe_${SLURM_JOB_ID}_${SLURM_NODEID}

if [ "${SLURM_LOCALID}" -eq 0 ]; then
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
    nvidia-cuda-mps-control -d
fi

# Apptainer forwards APPTAINERENV_* into the container.
export APPTAINERENV_CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY}
export APPTAINERENV_CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY}

# TALP (DLB): set TALP=1 in the environment to LD_PRELOAD libdlb_mpi.so and
# have it dump a per-run efficiency report; see analysis/talp_compare.py.
APPTAINER_ENV_ARGS=(--env UCX_POSIX_USE_PROC_LINK=n)
if [ "${TALP:-0}" = "1" ]; then
    APPTAINER_ENV_ARGS+=(
        --env LD_PRELOAD="/opt/dlb/lib/libdlb_mpi.so"
        --env DLB_ARGS="--talp --talp-output-file=ElmerIce_Talp_${TAG}_${SLURM_JOB_ID}.json"
    )
fi

exec apptainer exec --nv --bind="${BIND}" "${APPTAINER_ENV_ARGS[@]}" \
    ${CONTAINER} ElmerSolver_mpi $1
