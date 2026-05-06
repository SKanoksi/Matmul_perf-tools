#!/bin/bash
#SBATCH -p compute-devel       # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=4    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J matmul.mpi          # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} ***

module purge
module load cpeIntel/25.03
module load perftools-base perftools
#module load perftools-base perftools-lite

ulimit -s unlimited

MATMUL_BUILD_FLAGS="-O3 -DUSE_ALGOR=4 -DSELECT_BLAS=1 -DUSE_MPI=1 -DUSE_MPI_IO=1 -qmkl=sequential "
MATMUL_SRC_DIR=$(realpath ../src)

CC ${MATMUL_BUILD_FLAGS} -o ./matmul_mpi.exe -I${MATMUL_SRC_DIR} ${MATMUL_SRC_DIR}/matmul_main.cpp

pat_build -g mpi,lustre,io,sysio,dl -u -o ./matmul_mpi_pat.exe ./matmul_mpi.exe

srun ./matmul_mpi_pat.exe 100 10000 10000 10000

pat_report ./matmul_mpi_pat.exe+*
pat_report -s traced_functions=show ./matmul_mpi_pat.exe+* | tail -n25

rm -rf ./matmul_mpi.exe ./matmul_mpi_pat.exe*

