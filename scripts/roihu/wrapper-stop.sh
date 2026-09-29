#!/bin/bash
#
# Invoked once per MPI rank by run_mps.sh after the solver step; only the
# node-local rank 0 has a daemon to shut down.

export CUDA_MPS_LOG_DIRECTORY=./pipe_${SLURM_JOB_ID}_${SLURM_NODEID}
export CUDA_MPS_PIPE_DIRECTORY=./pipe_${SLURM_JOB_ID}_${SLURM_NODEID}

if [ "${SLURM_LOCALID}" -eq 0 ]; then
    echo quit | nvidia-cuda-mps-control
fi
