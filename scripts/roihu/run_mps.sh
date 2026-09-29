#!/bin/bash -l
# MESH LEVEL
export MESH_LEVEL=$1

# DIR PATHS
export BASEDIR="/scratch/project_2001659/danieree/rsync/my_ElmerIceEnergy"
export TAG="N${SLURM_NNODES}_n${SLURM_NTASKS_PER_NODE}_c${SLURM_CPUS_PER_TASK}_ML${MESH_LEVEL}_MPS"
export RUNDIR="${BASEDIR}/runs/roihu/${TAG}/run_Elmer_roihu_${TAG}_${SLURM_JOB_ID}"
export SCRIPTSDIR="${BASEDIR}/scripts"
export CONTAINERSDIR="${BASEDIR}/containers"
export INPUTSDIR="${BASEDIR}/inputs"

# OMPI SETTINGS
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

export BIND="$(csc-common-bind),${GREENLAND}"

# ELMERGRID -- partition count is the total rank count across all nodes.
srun -N1 -n1 apptainer exec --bind="${BIND}" ${CONTAINER} ElmerGrid 2 2 MESH -partdual -metiskway ${SLURM_NTASKS}

# ELMERF90
srun -N1 -n1 apptainer exec --bind="${BIND}" ${CONTAINER} elmerf90 Scalar_OUTPUT.F90 -o Scalar_OUTPUT

# ELMERSOLVER -- see wrapper-start.sh for the per-rank MPS setup.
start=$(date +%s)
srun -n ${SLURM_NTASKS} --cpu-bind=cores --cpus-per-task=${SLURM_CPUS_PER_TASK} ${SCRIPTSDIR}/roihu/wrapper-start.sh SSA_amgx_ML${MESH_LEVEL}.sif
srun -n ${SLURM_NTASKS} ${SCRIPTSDIR}/roihu/wrapper-stop.sh
end=$(date +%s)

echo "Elapsed time: $(($end-$start)) s"
echo "-----------------------------------"

rm -r ${GREENLAND}/MESH
