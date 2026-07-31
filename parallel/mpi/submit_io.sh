#!/bin/bash
#SBATCH -p compute             # Partition
#SBATCH -N 1                   # Number of node
#SBATCH --ntasks-per-node=64   # Number of MPI processes per node
#SBATCH --cpus-per-task=2      # Number of OpenMP threads per MPI process
#SBATCH -t 00:30:00            # Job runtime limit
#SBATCH -J matmul.par          # Job name
#SBATCH -A thaisc              # Slurm Account
#SBATCH --reservation=         # Reservation, if any
#SBATCH --array=0,1,2

jobs_input_args='20 12000 12000 12000'
jobs_header_dir=$(realpath ../../src)
jobs_src_dir=$(realpath ../../src)

matmul_build_flags="-DUSE_MPI=3 -DUSE_MPI_IO=${SLURM_ARRAY_TASK_ID} "
matmul_build_flags="${matmul_build_flags} -DUSE_OMP=1 -qopenmp" # -qmkl=parallel -qopenmp"
matmul_build_flags="-O3 -DUSE_ALGOR=3 -DWRITE_ARRAYS=1 -DNUM_ITER_PER_WRITE=4 ${matmul_build_flags}"

enable_perftools_trace=0
pat_build_opts="-g mpi,omp,pthreads,blas,lustre,io,sysio,dl -u "

# ---

#NUM_OST=8
#OST_HINT="striping_factor=${NUM_OST}"
#NUM_CB_PER_OST=4
#CB_HINT=":cray_cb_nodes_multiplier=${NUM_CB_PER_OST}"

#export MPICH_MPIIO_HINTS="*_*.bin:${OST_HINT}${CB_HINT}"
#export MPICH_MPIIO_HINTS_DISPLAY=1

# ---

export OMP_SCHEDULE=auto
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#export OMP_PLACES=cores
#export OMP_PROC_BIND=true

#export OMP_AFFINITY_FORMAT="Thread: %.4n of %.4N (affinity: %.10A) [%.6P : %.12H]"
#export OMP_DISPLAY_AFFINITY=true

# -----------------

module purge
module load cpeIntel/25.03

if [ ${enable_perftools_trace} -eq 1 ]; then
  module load perftools-base perftools
fi

mkdir mpiio_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
cd mpiio_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

CC ${matmul_build_flags} -o ./matmul_par.exe \
  -I${jobs_header_dir} ${jobs_src_dir}/matmul_main.cpp

if [ ${enable_perftools_trace} -eq 1 ]; then

  pat_build ${pat_build_opts} -o ./matmul_par_pat.exe ./matmul_par.exe

  export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_VEC_INS"
  export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_FP_INS,PAPI_FP_OPS"
  export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_BR_INS,PAPI_BR_MSP"

  srun -o slurm-%j_instr.out ./matmul_par_pat.exe ${jobs_input_args}

  pat_report ./matmul_par_pat.exe+* >& ./pat_instr.out
  pat_report -s traced_functions=show ./matmul_par_pat.exe+* | tail -n25 >> ./pat_instr.out

  rm -rf ./matmul_par_pat.exe+*

  # -----

  export PAT_RT_PERFCTR="PAPI_L1_DCA,PAPI_L1_DCM,PAPI_L2_DCM"
  export PAT_RT_PERFCTR="${PAT_RT_PERFCTR},PAPI_TOT_INS,PAPI_TLB_DM"

  srun -o slurm-%j_cache.out ./matmul_par_pat.exe ${jobs_input_args}

  pat_report ./matmul_par_pat.exe+* >& ./pat_cache.out
  pat_report -s traced_functions=show ./matmul_par_pat.exe+* | tail -n25 >> ./pat_cache.out

  rm -rf ./matmul_par.exe ./matmul_par_pat.exe*

else
  srun ./matmul_par.exe ${jobs_input_args}
fi


