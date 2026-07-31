#!/bin/bash
#SBATCH -p compute             # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -t 00:10:00            # Job runtime limit
#SBATCH -J PAPI                # Job name
#SBATCH -A thaisc              # Slurm Account 
#SBATCH --reservation=         # Reservation, if any

module reset
module swap PrgEnv-cray PrgEnv-gnu
module load perftools

CC -DUSE_ALGOR=0 -O3 -o ./matmul.exe -I../../src ../../src/matmul_main.cpp 

pat_build -o ./matmul_pat.exe ./matmul.exe

# ---
#
export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_VEC_INS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_FP_INS,PAPI_FP_OPS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_BR_INS,PAPI_BR_MSP"

srun ./matmul_pat.exe 20 1000 1000 1000

pat_report ./matmul_pat.exe+*-* >& pat_instr.out

rm -rf ./matmul_pat.exe+*-*

# ---

export PAT_RT_PERFCTR="PAPI_L1_DCA,PAPI_L1_DCM,PAPI_L2_DCM"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_TOT_INS,PAPI_TLB_DM"

srun ./matmul_pat.exe 20 1000 1000 1000

pat_report ./matmul_pat.exe+*-* >& pat_cache.out

rm -rf ./matmul.exe ./matmul_pat.exe*



