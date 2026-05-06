#!/bin/bash
#SBATCH -p compute-devel       # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J CrayPat             # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} ***

module reset
module load cray-python/3.11.7
module load perftools-base
module load perftools-preload

srun pat_run python3 ../src/matmul.py 25 1000 1000 1000

pat_report ./python3+*

#rm -rf ./python3+*
