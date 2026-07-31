#!/bin/bash
#SBATCH -p gpu-devel           # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH -t 00:30:00            # Job runtime limit
#SBATCH -J matmul.gpu          # Job name
#SBATCH -A thaisc              # Slurm Account
#SBATCH --reservation=         # Reservation, if any
#SBATCH --array=6,8,10,12,14,16  # --> Scale factor -> Matrix size

mat_size=$((1000*${SLURM_ARRAY_TASK_ID}))
jobs_input_args="40 ${mat_size} ${mat_size} ${mat_size} "

export MPICH_GPU_SUPPORT_ENABLED=0

MATMUL_BUILD_FLAGS="-DUSE_MPI=2 -DNUM_SPLIT_MPI_CALL=4 -DUSE_MPI_GPU_DIRECT=0 "
MATMUL_BUILD_FLAGS="-O3 -DUSE_ALGOR=4 -DSELECT_BLAS=2 ${MATMUL_BUILD_FLAGS}"
MATMUL_SRC_DIR=$(realpath ../../src)

enable_perftools_trace=1
pat_build_opts="-g cuda,cuda_math,blas,mpi,dl -u "

# -------------

mkdir matsize_${SLURM_ARRAY_JOB_ID}_${mat_size}
cd matsize_${SLURM_ARRAY_JOB_ID}_${mat_size}

module reset
module swap cray-libsci cray-libsci_acc
module load cuda/12.6
module load craype-accel-nvidia80

if [ ${enable_perftools_trace} -eq 1 ]; then
  module load perftools
fi

export LIBRARY_PATH=${LIBRARY_PATH}:${LD_LIBRARY_PATH}

CC ${MATMUL_BUILD_FLAGS} -o ./matmul_gpu.exe \
	-I${MATMUL_SRC_DIR} ${MATMUL_SRC_DIR}/matmul_main.cpp -lcurand 

if [ ${enable_perftools_trace} -eq 1 ]; then

  pat_build ${pat_build_opts} \
	  -o ./matmul_gpu_pat.exe ./matmul_gpu.exe

  srun ../dist_mpi2gpu ./matmul_gpu_pat.exe $(echo "${jobs_input_args}") 

  pat_report ./matmul_gpu_pat.exe+*
  pat_report -s traced_functions=show ./matmul_gpu_pat.exe+* | tail -n25

  rm -rf ./matmul_gpu.exe ./matmul_gpu_pat.exe*

else 

  srun ../dist_mpi2gpu ./matmul_gpu.exe $(echo "${jobs_input_args}")

fi


