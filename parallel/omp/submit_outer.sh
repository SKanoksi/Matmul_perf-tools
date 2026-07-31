#!/bin/bash
#SBATCH -p compute             # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=1    # Number of MPI processes per node
#SBATCH --cpus-per-task=64     # Number of OpenMP threads per MPI process 
#SBATCH --cores-per-socket=64
#SBATCH -t 00:20:00            # Job runtime limit
#SBATCH -J matmul.omp          # Job name
#SBATCH -A thaisc              # Slurm Account 
#SBATCH --reservation=         # Reservation, if any
#SBATCH --array=2,4,5,8,10,16,20,25,32    # --> Chunck size

module purge
module load cpeIntel/25.03

mkdir chunk_${SLURM_ARRAY_TASK_ID}
cd chunk_${SLURM_ARRAY_TASK_ID}

#export OMP_AFFINITY_FORMAT="Thread: %.4n of %.4N (affinity: %.10A) [%.6P : %.12H]"
#export OMP_DISPLAY_AFFINITY=true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_SCHEDULE=monotonic:static,${SLURM_ARRAY_TASK_ID}

export OMP_PLACES=cores
export OMP_PROC_BIND=true

jobs_input_args="200 8000 1 12800"
enable_perftools_trace=0
pat_build_opts="-g omp,pthreads,pthreads_mutex,dl"

# --- --- --- --- ---

if [ ${enable_perftools_trace} -eq 1 ]; then
  module load perftools-base perftools
fi

CC -O3 -DUSE_ALGOR=0 -DUSE_OMP=2 -DWRITE_ARRAYS=0 -DPRINT_LOOP_ITER=0 \
       	-o ./matmul_par.exe -I../../../src ../../../src/matmul_main.cpp -qopenmp

if [ ${enable_perftools_trace} -eq 1 ]; then

  pat_build ${pat_build_opts} -o ./matmul_par_pat.exe ./matmul_par.exe

  export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_VEC_INS"
  export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_FP_INS,PAPI_FP_OPS"
  export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_BR_INS,PAPI_BR_MSP"

  srun ./matmul_par_pat.exe $(echo ${jobs_input_args})

  pat_report ./matmul_par_pat.exe+* >& ./pat_instr.out
  pat_report -s traced_functions=show ./matmul_par_pat.exe+* | tail -n25 >> ./pat_instr.out
  rm -rf ./matmul_par_pat.exe+*

  # ---

  export PAT_RT_PERFCTR="PAPI_L1_DCA,PAPI_L1_DCM,PAPI_L2_DCM"
  export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_TOT_INS,PAPI_TLB_DM"

  srun ./matmul_par_pat.exe $(echo ${jobs_input_args}) 

  pat_report ./matmul_par_pat.exe+* >& ./pat_cache.out
  pat_report -s traced_functions=show ./matmul_par_pat.exe+* | tail -n25 >> ./pat_cache.out
  rm -rf ./matmul_par.exe ./matmul_par_pat.exe*

else
  srun ./matmul_par.exe $(echo ${jobs_input_args})
fi


