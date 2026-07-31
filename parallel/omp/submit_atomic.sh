#!/bin/bash
#SBATCH -p compute             # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=8      # Number of OpenMP threads per MPI process
#SBATCH --cores-per-socket=64 
#SBATCH -t 00:10:00            # Job runtime limit
#SBATCH -J matmul.omp          # Job name
#SBATCH -A thaisc              # Slurm Account
#SBATCH --reservation=         # Reservation, if any
#SBATCH --array=1,2            # --> USE_OMP=1 and 2

module purge
module load cpeIntel/25.03
module load perftools-base perftools

INPUT_ARGS='20 640 640 640'

#export OMP_AFFINITY_FORMAT="Thread: %.4n of %.4N (affinity: %.10A) [%.6P : %.12H]"
#export OMP_DISPLAY_AFFINITY=true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=true

# ---

mkdir omp${SLURM_ARRAY_TASK_ID}
cd omp${SLURM_ARRAY_TASK_ID}

CC -O3 -DUSE_ALGOR=1 -DUSE_OMP=${SLURM_ARRAY_TASK_ID} \
       	-o ./matmul.exe -I../../../src ../../../src/matmul_main.cpp -qopenmp

pat_build -g omp,pthreads,pthreads_mutex,dl -u \
	-o ./matmul_pat.exe ./matmul.exe

# ---

export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_VEC_INS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_FP_INS,PAPI_FP_OPS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_BR_INS,PAPI_BR_MSP"

srun ./matmul_pat.exe $(echo ${INPUT_ARGS})

pat_report ./matmul_pat.exe+*-* >& pat_instr.out

rm -rf ./matmul_pat.exe+*-*

# ---

export PAT_RT_PERFCTR="PAPI_L1_DCA,PAPI_L1_DCM,PAPI_L2_DCM"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_TOT_INS,PAPI_TLB_DM"

srun ./matmul_pat.exe $(echo ${INPUT_ARGS})

pat_report ./matmul_pat.exe+*-* >& pat_cache.out

rm -rf ./matmul.exe ./matmul_pat.exe*



