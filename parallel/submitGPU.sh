#!/bin/bash
#SBATCH -p gpu-devel           # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J matmul.gpu          # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} ***

module reset
module swap cray-libsci cray-libsci_acc
module load cuda/12.6
module load craype-accel-nvidia80
module load perftools

ulimit -s unlimited

MATMUL_BUILD_FLAGS="-O3 -DUSE_ALGOR=4 -DSELECT_BLAS=2 "
MATMUL_SRC_DIR=$(realpath ../src)

CC ${MATMUL_BUILD_FLAGS} -o ./matmul_gpu.exe -I${MATMUL_SRC_DIR} ${MATMUL_SRC_DIR}/matmul_main.cpp

pat_build -g cuda,cuda_math,mpi,dl -u -o ./matmul_gpu_pat.exe ./matmul_gpu.exe

srun ./matmul_gpu_pat.exe 100 10000 10000 10000

pat_report ./matmul_gpu_pat.exe+*
pat_report -s traced_functions=show ./matmul_gpu_pat.exe+* | tail -n25

rm -rf ./matmul_gpu.exe ./matmul_gpu_pat.exe*
