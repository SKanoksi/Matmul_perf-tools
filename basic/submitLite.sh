#!/bin/bash
#SBATCH -p compute-devel       # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J CrayPat             # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} *** 

module reset
module swap PrgEnv-cray PrgEnv-gnu
module load perftools-lite

CC -o ./matmul_lite.exe -I../src ../src/matmul_main.cpp 

srun ./matmul_lite.exe 25 1000 1000 1000

pat_report ./matmul_lite.exe+*-*

#rm -rf ./matmul_lite.exe*

