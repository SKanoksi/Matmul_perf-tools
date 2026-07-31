#!/bin/bash
#SBATCH -p compute             # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=1      # Number of OpenMP threads per MPI process
#SBATCH -B 1:64:1
#SBATCH -t 00:20:00            # Job runtime limit
#SBATCH -J matmul              # Job name
#SBATCH -A thaisc              # Slurm Account 
#SBATCH --reservation=         # Reservation, if any

module purge
module load cpeIntel/25.03
module load perftools-base perftools

CC -O3 -DUSE_ALGOR=2 \
	-DWRITE_ARRAYS=0 \
	-o ./matmul.exe -I. ../../src/matmul_main.cpp

pat_build -o ./matmul_pat.exe ./matmul.exe

# -----

export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_VEC_INS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_FP_INS,PAPI_FP_OPS"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_BR_INS,PAPI_BR_MSP"

echo "--> PAT_RT_PERFCTR=${PAT_RT_PERFCTR} <---" >& tune_program.out

srun ./matmul_pat.exe >> tune_program.out
pat_report ./matmul_pat.exe+* >& tune_pat_inst.out

rm -rf ./matmul_pat.exe_craypat_instr
mv ./matmul_pat.exe+* ./matmul_pat.exe_craypat_instr

# -----

export PAT_RT_PERFCTR="PAPI_L1_DCA,PAPI_L1_DCM,PAPI_L2_DCM"
export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_TOT_INS,PAPI_TLB_DM"

echo "--> PAT_RT_PERFCTR=${PAT_RT_PERFCTR} <---" >> tune_program.out

srun ./matmul_pat.exe >> tune_program.out
pat_report ./matmul_pat.exe+* >& tune_pat_cache.out

rm -rf ./matmul_pat.exe_craypat_cache
mv ./matmul_pat.exe+* ./matmul_pat.exe_craypat_cache

