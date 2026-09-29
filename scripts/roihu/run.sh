#!/bin/bash -l

# MESH LEVEL
export MESH_LEVEL=$1

# DIR PATHS
export BASEDIR="/scratch/project_2001659/danieree/rsync/my_ElmerIceEnergy"
export TAG="N${SLURM_NNODES}_n${SLURM_NTASKS_PER_NODE}_c${SLURM_CPUS_PER_TASK}_ML${MESH_LEVEL}"
export RUNDIR="${BASEDIR}/runs/roihu/${TAG}/run_Elmer_roihu_${TAG}_${SLURM_JOB_ID}"
export SCRIPTSDIR="${BASEDIR}/scripts"
export CONTAINERSDIR="${BASEDIR}/containers"
export INPUTSDIR="${BASEDIR}/inputs"

# OMPI SETTINGS (run_test.sh has these commented out; Leonardo runs with them on)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PMIX_MCA_gds=hash
export PMIX_MCA_psec=native
export OMPI_MCA_btl=^openib

# CONTAINER PATH
export CONTAINER=${CONTAINERSDIR}/container.sif
export GREENLAND=${RUNDIR}/Greenland_SSA

# PREPROCESS
mkdir -p ${RUNDIR}
tar -xvzf "${INPUTSDIR}/Greenland_SSA.tar.gz" -C ${RUNDIR}
cd ${GREENLAND}

# Resolved once rather than per rank -- it forks a subprocess.
export BIND="$(csc-common-bind),${GREENLAND}"

# ELMERGRID -- partition count is the total rank count across all nodes.
srun -N1 -n1 apptainer exec --bind="${BIND}" ${CONTAINER} ElmerGrid 2 2 MESH -partdual -metiskway ${SLURM_NTASKS}

# ELMERF90
srun -N1 -n1 apptainer exec --bind="${BIND}" ${CONTAINER} elmerf90 Scalar_OUTPUT.F90 -o Scalar_OUTPUT

# TALP (DLB): set TALP=1 in the environment to LD_PRELOAD libdlb_mpi.so and
# have it dump a per-run efficiency report; see analysis/talp_compare.py.
export DLB_PREFIX="/opt/dlb"
APPTAINER_ENV_ARGS=(--env UCX_POSIX_USE_PROC_LINK=n)
if [ "${TALP:-0}" = "1" ]; then
    APPTAINER_ENV_ARGS+=(
        --env LD_PRELOAD="${DLB_PREFIX}/lib/libdlb_mpi.so"
        --env DLB_ARGS="--talp --talp-output-file=ElmerIce_Talp_${TAG}_${SLURM_JOB_ID}.json"
    )
fi

start=$(date +%s)
srun -n ${SLURM_NTASKS} --cpu-bind=cores --cpus-per-task=${SLURM_CPUS_PER_TASK} apptainer exec --nv --bind="${BIND}" "${APPTAINER_ENV_ARGS[@]}" ${CONTAINER} ElmerSolver_mpi SSA_amgx_ML${MESH_LEVEL}.sif
end=$(date +%s)

echo "Elapsed time: $(($end-$start)) s"
echo "-----------------------------------"

# Check the results
rm -r ${GREENLAND}/MESH
