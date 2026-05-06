#!/bin/bash
#SBATCH -p compute-devel       # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=4      # Number of OpenMP threads per MPI process
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J matmul.omp          # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} ***

module purge
module load cpeIntel/25.03
module load perftools-base perftools

ulimit -s unlimited
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_SCHEDULE=auto

MATMUL_BUILD_FLAGS="-O3 -DUSE_ALGOR=4 -DUSE_OMP=1 -DSELECT_BLAS=1 -qopenmp -qmkl=parallel "
MATMUL_SRC_DIR=$(realpath ../src)

CC ${MATMUL_BUILD_FLAGS} -o ./matmul_omp.exe -I${MATMUL_SRC_DIR} ${MATMUL_SRC_DIR}/matmul_main.cpp

pat_build -g omp,pthreads,blas,dl -u -o ./matmul_omp_pat.exe ./matmul_omp.exe

srun ./matmul_omp_pat.exe 100 10000 10000 10000

pat_report ./matmul_omp_pat.exe+*
pat_report -s traced_functions=show ./matmul_omp_pat.exe+* | tail -n25

rm -rf ./matmul_omp.exe ./matmul_omp_pat.exe*

