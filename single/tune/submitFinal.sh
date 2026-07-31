#!/bin/bash
#SBATCH -p compute             # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -B 1:64:1
#SBATCH -t 00:30:00            # Job runtime limit
#SBATCH -J matmul              # Job name
#SBATCH -A thaisc              # Slurm Account 
#SBATCH --reservation=         # Reservation, if any

module purge
module load cpeIntel/25.03

INPUT_ARGS='20 1080 1000 1120'
#INPUT_ARGS='50 3000 3000 3200'

echo "=================="
echo "*** MKL Ref ***"

CC -O3 -DUSE_ALGOR=4 -DSELECT_BLAS=1 \
	-DWRITE_ARRAYS=0 -DPRINT_LOOP_ITER=0 \
	-o ./matmul_mkl.exe -I. ../../src/matmul_main.cpp -qmkl=sequential

srun ./matmul_mkl.exe $(echo ${INPUT_ARGS})

echo ""
echo "=================="
echo "*** Libsci Ref ***"

module load cray-libsci

CC -O3 -DUSE_ALGOR=4 -DSELECT_BLAS=0 \
	-DWRITE_ARRAYS=0 -DPRINT_LOOP_ITER=0 \
        -o ./matmul_libsci.exe -I. ../../src/matmul_main.cpp

srun ./matmul_libsci.exe $(echo ${INPUT_ARGS})

echo ""
echo "=================="
echo "*** My Final ***"

CC -O3 -DUSE_ALGOR=3 \
        -DWRITE_ARRAYS=0 -DPRINT_LOOP_ITER=0 \
        -o ./matmul.exe -I. ../../src/matmul_main.cpp

srun ./matmul.exe $(echo ${INPUT_ARGS})

rm -rf ./matmul*.exe

