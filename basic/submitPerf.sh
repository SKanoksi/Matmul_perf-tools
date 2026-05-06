#!/bin/bash
#SBATCH -p compute-devel       # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J CrayPat             # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} *** 

module reset
module load perftools

CC -o ./matmul.exe -I../src ../src/matmul_main.cpp 

pat_build -o ./matmul_pat.exe ./matmul.exe

srun ./matmul_pat.exe 25 1000 1000 1000

pat_report ./matmul_pat.exe+*-*

#rm -rf ./matmul.exe ./matmul_pat.exe*

