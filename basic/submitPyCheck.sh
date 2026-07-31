#!/bin/bash
#SBATCH -p memory              # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J mm.check            # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} *** 

ml reset
ml cray-python

srun python3 ../src/check.py 25 1000 1000 1000 10

