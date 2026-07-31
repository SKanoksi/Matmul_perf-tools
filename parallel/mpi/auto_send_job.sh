#!/bin/bash

# Suggestion :: Use command such as
#   grep 'Time :: Main loop' ./N*/matmul*.out 
#   tail ./N*/matmul*.out
# to inspect the logs at the same time


jobs_input_args='50 4000 4000 4000'
#jobs_input_args='20 12000 12000 12000'

jobs_prefix_dir="PureMPI-3_"
jobs_num_task=("1" "2" "4" "8" "16" "32" "64")    # Here, num task == MPI size 
jobs_num_thread=("1" "1" "1" "1" "1" "1" "1")

matmul_build_flags="-O3 -DUSE_ALGOR=3 -DUSE_MPI=3 "
matmul_build_flags="${matmul_build_flags} -DUSE_OMP=0 -qopenmp" # -qmkl=parallel -qopenmp"
matmul_build_flags="${matmul_build_flags} -DWRITE_ARRAYS=0 -DUSE_MPI_IO=0 "

enable_perftools_trace=1
pat_build_opts="-g mpi,omp,pthreads,blas,lustre,io,sysio,dl -u "

account="thaisc"
reservation=""
partition="compute"
max_core_per_node=128

jobs_header_dir=$(realpath ../../src)
jobs_src_dir=$(realpath ../../src)


# -------- END OF SETUP SECTION --------


function min(){ 
  echo $(( $1 < $2 ? $1 : $2 )) 
}

for ((i=0 ; i<$(min ${#jobs_num_task[@]} ${#jobs_num_thread[@]}) ; i++))
do

  ntask=${jobs_num_task[$i]}
  nthread=${jobs_num_thread[$i]}
  nnode=$((1+(${ntask}*${nthread}-1)/${max_core_per_node}))

  if [ ${nnode} -gt ${ntask} ]; then
    echo "Error -- Calculated num node (${nnode}) > Specified ntask (${ntask})"
    continue 
  fi
  ntaskpnode=$(((${ntask}+${nnode}-1)/${nnode}))

  workdirname="${jobs_prefix_dir}N${nnode}n${ntask}c${nthread}"
  rm -rf ${workdirname}
  mkdir ${workdirname}

  echo "Sending $((${i}+1)) job -- using ${ntask} tasks and ${nthread} threads."

#cat << Eof
sbatch -D ${workdirname} << Eof
#!/bin/bash
#SBATCH -p ${partition}
#SBATCH -N ${nnode}
#SBATCH --ntasks=${ntask}
#SBATCH --ntasks-per-node=${ntaskpnode}
#SBATCH --cpus-per-task=${nthread}
#SBATCH --cores-per-socket=64
#SBATCH -t 00:30:00             
#SBATCH -J matmul.par
#SBATCH --output=slurm-%j_N${nnode}n${ntask}c${nthread}.out
#SBATCH -A ${account}
#SBATCH --reservation=${reservation}

module purge
module load cpeIntel/25.03

if [ ${enable_perftools_trace} -eq 1 ]; then
  module load perftools-base perftools
fi

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OMP_SCHEDULE=auto

CC ${matmul_build_flags} -o ./matmul_par.exe \
  -I${jobs_header_dir} ${jobs_src_dir}/matmul_main.cpp

if [ ${enable_perftools_trace} -eq 1 ]; then

  pat_build ${pat_build_opts} -o ./matmul_par_pat.exe ./matmul_par.exe

  export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_VEC_INS"
  export PAT_RT_PERFCTR="\${PAT_RT_PERFCTR},PAPI_FP_INS,PAPI_FP_OPS"
  export PAT_RT_PERFCTR="\${PAT_RT_PERFCTR},PAPI_BR_INS,PAPI_BR_MSP" 

  srun -o slurm-%j_instr.out ./matmul_par_pat.exe ${jobs_input_args}

  pat_report ./matmul_par_pat.exe+* >& ./pat_instr.out
  pat_report -s traced_functions=show ./matmul_par_pat.exe+* | tail -n25 >> ./pat_instr.out

  rm -rf ./matmul_par_pat.exe+*

  # -----
  
  export PAT_RT_PERFCTR="PAPI_L1_DCA,PAPI_L1_DCM,PAPI_L2_DCM"
  export PAT_RT_PERFCTR="\${PAT_RT_PERFCTR},PAPI_TOT_INS,PAPI_TLB_DM"

  srun -o slurm-%j_cache.out ./matmul_par_pat.exe ${jobs_input_args} 

  pat_report ./matmul_par_pat.exe+* >& ./pat_cache.out
  pat_report -s traced_functions=show ./matmul_par_pat.exe+* | tail -n25 >> ./pat_cache.out

  rm -rf ./matmul_par.exe ./matmul_par_pat.exe*

else
  srun ./matmul_par.exe ${jobs_input_args}
fi

Eof

  # To be gentle with SLURM, avoid DDoS attack it
  sleep 1.0

done


