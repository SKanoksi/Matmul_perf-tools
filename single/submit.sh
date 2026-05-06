#!/bin/bash
#SBATCH -p compute-devel       # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -t 02:00:00            # Job runtime limit
#SBATCH -J matmul              # Job name
#SBATCH -A thaisc              # Account *** {USER EDIT} *** 

module purge
module load cpeGNU/25.03
module load perftools-base perftools

CC -O3 -DUSE_ALGOR=2 -DUSE_PAT_API=2 -DWRITE_ARRAYS=0 \
	-o ./matmul.exe -I../src ../src/matmul_main.cpp

pat_build -w -o ./matmul_pat.exe ./matmul.exe

export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_VEC_INS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_FP_INS,PAPI_FP_OPS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_BR_INS,PAPI_BR_MSP"

#export PAT_RT_PERFCTR="PAPI_L1_DCA,PAPI_L1_DCM,PAPI_L2_DCM"
#export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_TOT_INS,PAPI_TLB_DM"

srun ./matmul_pat.exe 10 

pat_report ./matmul_pat.exe+*

rm -rf ./matmul.exe ./matmul_pat.exe*

