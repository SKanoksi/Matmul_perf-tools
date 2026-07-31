/******************************************************************\

  Matmul -- Perf. tools

  Version 2.0.0
  Copyright (c) 2026, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.

\******************************************************************/

#include <matmul_setup.hpp>

#if USE_MPI>0
  #include <mpi.h>
#endif

#if USE_OMP>0
  #include <omp.h>
#endif

#if defined(CRAYPAT) && USE_PAT_API>0
  #include <pat_api.h>
#endif

#if USE_MPI>2 || (USE_MPI>0 && WRITE_ARRAYS>0 && USE_MPI_IO<1)
  #include <vector>
#endif

#include "matmul_util.hpp"
#include "matmul_algor.hpp"


static_assert(std::is_floating_point<Float>::value,
              "Float must be a floating point datatype."
              );

#if USE_MPI_IO>0
template <typename U, MPI_Datatype mpi_type>
void write_array_mpi(const U *A,
                     const std::size_t offset, const std::size_t size,
                     const std::string filename)
{
  MPI_File fh ;
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);

  MPI_Offset mpi_offset = (MPI_Offset) offset * sizeof(U);

#if USE_MPI_IO>1
  MPI_File_write_at_all(fh, mpi_offset, A, size,
                        mpi_type, MPI_STATUS_IGNORE);
#else
  MPI_File_write_at(fh, mpi_offset, A, size,
                    mpi_type, MPI_STATUS_IGNORE);
#endif
  MPI_File_close(&fh);

return; }
#endif


int main(int argc, char *argv[])
{
#if USE_MPI>0 // Init MPI

#if USE_OMP>0
  int mpi_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided);
  if( mpi_provided < MPI_THREAD_FUNNELED )
  {
    std::cerr << "The MPI_THREAD_FUNNELED is NOT supported." << std::endl;
    //MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
  }
  int num_threads = 1 ;
#else
  MPI_Init(&argc, &argv);
#endif

  int mpi_rank, mpi_size ;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
  constexpr int mpi_rank = 0, mpi_size = 1 ;
  int num_threads = 1 ;
#endif // USE_MPI>0 -- Init MPI

  reset_clock(mpi_rank, clock_type::loop);
  start_clock(mpi_rank, clock_type::prog);

  int num_repeat = DEFAULT_NUM_REPEAT ;
  int m_size = DEFAULT_MATRIX_M_SIZE ;
  int n_size = DEFAULT_MATRIX_N_SIZE ;
  int p_size = DEFAULT_MATRIX_P_SIZE ;
  if( argc>1 ){
    num_repeat = std::atoi(argv[1]) ;
  }
  if( argc>2 ){
    m_size = std::atoi(argv[2]) ;
    m_size = (m_size<0) ? -m_size : +m_size ;
  }
  if( argc>3 ){
    n_size = std::atoi(argv[3]) ;
    n_size = (n_size<0) ? -n_size : +n_size ;
  }
  if( argc>4 ){
    p_size = std::atoi(argv[4]) ;
    p_size = (p_size<0) ? -p_size : +p_size ;
  }
  if( argc>5 ){
    std::cerr
    << "Incorrect input argument -- "
    << "Only num_repeat, m_size, n_size and p_size (4 arguments) are accepted."
    << std::endl;
#if USE_MPI>0
    //MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    return 1;
  }

#if USE_MPI>0
  if( mpi_rank == 0 ){
    std::cout 
    << "Using " << mpi_size << " MPI processes --> " 
#if USE_MPI==1
    << "Blocking BCAST"
#elif USE_MPI==2
    << "Non-Blocking BCAST"
#else
    << "Non-Blocking ALLGATHER"
#endif
    << std::endl;
#else
  {
    std::cout << "Without MPI" << std::endl;
#endif
#if USE_OMP>0
  #pragma omp parallel
  {
    if( omp_get_thread_num()==0 ){
      num_threads = omp_get_num_threads();
      std::cout 
      << "Using " << num_threads << " OMP threads --> " 
#if USE_OMP==1
      << "omp parallel for"
#else
      << "omp parallel for collapse(n)"
#endif
      << std::endl;
    }
  }
#else
  std::cout << "Without OpenMP" << std::endl;
#endif
  }
 
#if USE_ALGOR==4 && SELECT_BLAS==2
  libsci_acc_init();
  if( mpi_rank == 0 )
    std::cout << "Using GPU threads" << std::endl;
#endif

  // ---

  int start_row_m, num_row_m, stripe_size_m, remaining_m ;
#if USE_MPI>0 && (USE_ALGOR==2 || USE_ALGOR==3)
  if( m_size % BLOCK_M_SIZE != 0 )
  {
    if( mpi_rank == 0 ){
      std::cerr
      << "\nINPUT_ERROR :: "
      << "matrix_m_size must be divisible by BLOCK_M_SIZE.\n"
      << std::endl;
    }
    //MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    return 1;
  }

  {
    int start_block_m, num_block_m, stripe_block_m, remaining_block_m ; 
    partition_dim(start_block_m, num_block_m, stripe_block_m, remaining_block_m,
                  mpi_rank, mpi_size, m_size/BLOCK_M_SIZE);
    
    start_row_m   =     start_block_m * BLOCK_M_SIZE ;
    num_row_m     =       num_block_m * BLOCK_M_SIZE ;
    stripe_size_m =    stripe_block_m * BLOCK_M_SIZE ;
    remaining_m   = remaining_block_m * BLOCK_M_SIZE ;
  }
#else
  partition_dim(start_row_m, num_row_m, stripe_size_m, remaining_m,
		mpi_rank, mpi_size, m_size);
#endif

#if USE_MPI>0 && WRITE_ARRAYS>0 && USE_MPI_IO<1
  std::vector<int> mpi_aa_recv_size(mpi_size), mpi_cc_recv_size(mpi_size) ;
  std::vector<int> mpi_aa_recv_disp(mpi_size), mpi_cc_recv_disp(mpi_size) ;

  if( mpi_rank==0 )
  {
    MPI_Gather(&num_row_m, 1, MPI_INT, mpi_aa_recv_size.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&start_row_m, 1, MPI_INT, mpi_aa_recv_disp.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i=0 ; i<mpi_size ; ++i)
    {
      mpi_cc_recv_size[i] = mpi_aa_recv_size[i] * p_size ;
      mpi_cc_recv_disp[i] = mpi_aa_recv_disp[i] * p_size ;

      mpi_aa_recv_size[i] *= n_size ;
      mpi_aa_recv_disp[i] *= n_size ;
    }

  }else{
    MPI_Gather(&num_row_m, 1, MPI_INT, nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&start_row_m, 1, MPI_INT, nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
#endif

  // ---

  int start_row_n = 0, num_row_n = n_size, stripe_size_n = n_size, remaining_n = 0 ;
#if USE_MPI>2 || USE_MPI_IO>0
  partition_dim(start_row_n, num_row_n, stripe_size_n, remaining_n,
                mpi_rank, mpi_size, n_size);
#endif

  const int rand_B_start_index = (USE_MPI>2) ? start_row_n*p_size : 0  ;
  const int rand_B_num_row = (USE_MPI>2) ? num_row_n : n_size ;

#if USE_MPI>0 && USE_MPI<3 && defined(NUM_SPLIT_MPI_CALL)
  constexpr int mpi_split_num_call = (int(NUM_SPLIT_MPI_CALL)<1) ? 1 : int(NUM_SPLIT_MPI_CALL) ;
  int mpi_split_Bsize = (rand_B_num_row*p_size)/mpi_split_num_call ;
  mpi_split_Bsize = (mpi_split_Bsize*mpi_split_num_call < rand_B_num_row*p_size) ? mpi_split_Bsize+1 : mpi_split_Bsize ;
#else
  constexpr int mpi_split_num_call = 1 ;
  int mpi_split_Bsize = rand_B_num_row*p_size ;
#endif

#if USE_MPI>2
  std::vector<int> mpi_bb_recv_size(mpi_size), mpi_bb_recv_disp(mpi_size) ;

  MPI_Allgather(&num_row_n, 1, MPI_INT, mpi_bb_recv_size.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&start_row_n, 1, MPI_INT, mpi_bb_recv_disp.data(), 1, MPI_INT, MPI_COMM_WORLD);

  for(int i=0 ; i<mpi_size ; ++i)
  {
    mpi_bb_recv_size[i] *= p_size ;
    mpi_bb_recv_disp[i] *= p_size ;
  }
#endif

  // ---
  
  if( mpi_rank == 0 )
  {
    std::cout
    << " Global matrix size: "
    << "[" << m_size << "," << p_size << "] = ["
    << m_size << "," << n_size << "] x ["
    << n_size << "," << p_size << "]"
    << std::endl;
#if USE_MPI>0
    std::cout
    << "  Local matrix size: "
    << "[" << stripe_size_m << "," << p_size << "] = ["
    << stripe_size_m << "," << n_size << "] x ["
    << n_size << "," << p_size << "]"
    << " <-- MPI"
    << std::endl;
#endif
#if USE_ALGOR==2 || USE_ALGOR==3
    std::cout
    << "  Matrix block size: "
    << "[" << BLOCK_M_SIZE << "," << BLOCK_P_SIZE << "] = ["
    << BLOCK_M_SIZE << "," << BLOCK_N_SIZE << "] x ["
    << BLOCK_N_SIZE << "," << BLOCK_P_SIZE << "]"
    << std::endl;
#endif
#if USE_ALGOR==3
    std::cout
    << "  Micro-kernel size: "
    << "[" << MICRO_M_SIZE << "," << MICRO_P_SIZE << "] = ["
    << MICRO_M_SIZE << ",1] x [1," << MICRO_P_SIZE << "]"
    << std::endl;
#endif
    std::cout
    << "--> Algorithm "
#if USE_ALGOR==1
    << "[1] = Loop interchange"
#elif USE_ALGOR==2
    << "[2] = Loop interchange + Cache blocking (no packing)"
#elif USE_ALGOR==3
    << "[3] = Micro kernel + Cache blocking (packing)"
#if defined(ALGOR2_NO_MERGE_LOOP) && ALGOR2_NO_MERGE_LOOP>0
    << " -- NO MERGE LOOP"
#endif
#elif USE_ALGOR==4
#if SELECT_BLAS==1
    << "[4] = BLAS lvl3 from <mkl.h>"
#elif SELECT_BLAS==2
    << "[4] = BLAS lvl3 from <libsci_acc.h>" 
#if defined(USE_MPI_GPU_DIRECT) && USE_MPI_GPU_DIRECT!=0
    << " -- GPUDirect"
#endif
#else
    << "[4] = BLAS lvl3 from <cblas.h>"
#endif
#else
    << "[0] = Trivial loops"
#endif
    << "\n" << std::endl;

    reset_clock(mpi_rank, clock_type::loop);
  }


#if USE_ALGOR==2 || USE_ALGOR==3
  if( num_row_m % BLOCK_M_SIZE != 0 )
  {
    std::cerr
    << "\nINPUT_ERROR :: "
#if USE_MPI<1
    << "m_size must be divisible by BLOCK_M_SIZE.\n"
#else
    // Actually, should exit since above
    << "[" << num_row_m << " % " << BLOCK_M_SIZE << " != 0] <-- "
    << "num_row_m of rank " << mpi_rank << " must be divisible by BLOCK_M_SIZE.\n"
#endif
    << std::endl;

#if USE_MPI>0
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }

  if( n_size % BLOCK_N_SIZE != 0 )
  {
    if( mpi_rank == 0 ){
      std::cerr
      << "\nINPUT_ERROR :: "
      << "matrix_n_size must be divisible by BLOCK_N_SIZE.\n"
      << std::endl;
    }
#if USE_MPI>0
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }

  if( p_size % BLOCK_P_SIZE != 0 )
  {
    if( mpi_rank == 0 ){
      std::cerr
      << "\nINPUT_ERROR :: "
      << "matrix_p_size must be divisible by BLOCK_P_SIZE.\n"
      << std::endl;
    }
#if USE_MPI>0
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }
#endif

  // ------------------------------

  const int alloc_size_m = (USE_MPI_IO==0 && mpi_rank==0) ? m_size : num_row_m ;
  
  Float *AA = (Float*)std::aligned_alloc(SIMD_ALIGNED_BYTE,
                                         alloc_size_m*n_size * sizeof(Float));
  Float *BB = (Float*)std::aligned_alloc(SIMD_ALIGNED_BYTE,
                                               n_size*p_size * sizeof(Float));
  Float *CC = (Float*)std::aligned_alloc(SIMD_ALIGNED_BYTE,
                                         alloc_size_m*p_size * sizeof(Float));
#if USE_ALGOR==4 && SELECT_BLAS==2
  Float *AA_device, *BB_device, *CC_device ;
  libsci_acc_DeviceAlloc((void **)&AA_device, alloc_size_m*n_size * sizeof(Float));
  libsci_acc_DeviceAlloc((void **)&BB_device,       n_size*p_size * sizeof(Float));
  libsci_acc_DeviceAlloc((void **)&CC_device, alloc_size_m*p_size * sizeof(Float));
#endif
  

#if USE_MPI>0 
#if WRITE_ARRAYS>0 && USE_MPI_IO<1
  if( stripe_size_m*n_size * sizeof(Float) > 1024*1024*1024 && mpi_rank == 0 )
  {
      std::cout << "\n Warning :: your A matrix may be too large for Gather.\n" << std::endl;
  }
#endif
#if USE_MPI>2
  if( mpi_split_Bsize * sizeof(Float) > 1024*1024*1024 && mpi_rank == 0 )
  {
      std::cout << "\n Warning :: your B matrix may be too large for Allgather.\n" << std::endl;
  }
#else
  if( mpi_split_Bsize * sizeof(Float) > 1024*1024*1024 && mpi_rank == 0 )
  {
      std::cout << "\n Warning :: your B matrix may be too large for Bcast.\n" << std::endl;
  }
#endif
#if WRITE_ARRAYS>0 && USE_MPI_IO<1
  if( stripe_size_m*p_size * sizeof(Float) > 1024*1024*1024 && mpi_rank == 0 )
  {
      std::cout << "\n Warning :: your C matrix may be too large for Gather.\n" << std::endl;
  }
#endif
#endif

  init_random_gen(mpi_rank);

#if USE_MPI>0
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // ------------------------------

  start_clock(mpi_rank, clock_type::loop);

#if defined(CRAYPAT) && USE_PAT_API>0
  PAT_region_begin(1,"iteration_loop");
#endif

#if USE_MPI>1
  MPI_Request  BB_request[mpi_split_num_call] ;
#endif

  // Main iteration loop == REPEAT
  for(int niter=0 ; niter < num_repeat ; ++niter)
  {
    if( mpi_rank == 0 ){  
#if PRINT_LOOP_ITER>0
      std::cout << "Begin loop " << niter+1 << "/" << num_repeat << std::endl;
#endif

#if USE_MPI>2
    }
#endif
#if USE_ALGOR!=4 || SELECT_BLAS!=2
      init_random<Float>(&BB[rand_B_start_index], rand_B_num_row*p_size);
#else 
      init_random<Float>(&BB_device[rand_B_start_index], rand_B_num_row*p_size);
#if USE_MPI>0 && (!defined(USE_MPI_GPU_DIRECT) || USE_MPI_GPU_DIRECT==0)
      libsci_acc_Memcpy(&BB[rand_B_start_index], &BB_device[rand_B_start_index], 
		        rand_B_num_row*p_size * sizeof(Float), libsci_acc_MemcpyDTH);
#endif
#endif
#if USE_MPI<3
    }
#endif

#if USE_MPI>0
    {
#if USE_MPI<3  // ###

      for(int i=0 ; i<mpi_split_num_call ; ++i)
      {
        const int start_index = i*mpi_split_Bsize ;
        const int msg_size = (start_index+mpi_split_Bsize < rand_B_num_row*p_size) ? mpi_split_Bsize : (rand_B_num_row*p_size) - start_index ;

#if USE_MPI==1
#if USE_ALGOR!=4 || SELECT_BLAS!=2 || !defined(USE_MPI_GPU_DIRECT) || USE_MPI_GPU_DIRECT==0
        MPI_Bcast(&BB[start_index], msg_size, CUSTOM_MPI_FLOAT, 0, MPI_COMM_WORLD);
#else
	MPI_Bcast(&BB_device[start_index], msg_size, CUSTOM_MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
#elif USE_MPI>1
#if USE_ALGOR!=4 || SELECT_BLAS!=2 || !defined(USE_MPI_GPU_DIRECT) || USE_MPI_GPU_DIRECT==0
        MPI_Ibcast(&BB[start_index], msg_size, CUSTOM_MPI_FLOAT, 0, MPI_COMM_WORLD, &BB_request[i]);
#else
        MPI_Ibcast(&BB_device[start_index], msg_size, CUSTOM_MPI_FLOAT, 0, MPI_COMM_WORLD, &BB_request[i]);
#endif
#endif // USE_MPI==1, USE_MPI>1
     }

#else // USE_MPI<3  ###
      
     // Cannot use with -DNUM_SPLIT_MPI_CALL != 1 
#if USE_ALGOR!=4 || SELECT_BLAS!=2 || !defined(USE_MPI_GPU_DIRECT) || USE_MPI_GPU_DIRECT==0
     MPI_Iallgatherv(MPI_IN_PLACE, mpi_bb_recv_size[mpi_rank], CUSTOM_MPI_FLOAT,
                     &BB[0], mpi_bb_recv_size.data(), mpi_bb_recv_disp.data(), CUSTOM_MPI_FLOAT,
                     MPI_COMM_WORLD, &BB_request[0]);
#else
     MPI_Iallgatherv(MPI_IN_PLACE, mpi_bb_recv_size[mpi_rank], CUSTOM_MPI_FLOAT,
                     &BB_device[0], mpi_bb_recv_size.data(), mpi_bb_recv_disp.data(), CUSTOM_MPI_FLOAT,
                     MPI_COMM_WORLD, &BB_request[0]);
#endif

#endif // USE_MPI<3 ###
    }
#endif

#if USE_MPI>0 && USE_MPI<2 && USE_ALGOR==4 && SELECT_BLAS==2 && (!defined(USE_MPI_GPU_DIRECT) || USE_MPI_GPU_DIRECT==0)
    libsci_acc_Memcpy(&BB_device[rand_B_start_index], &BB[rand_B_start_index], 
		      rand_B_num_row*p_size * sizeof(Float), libsci_acc_MemcpyHTD);
    // BB == data_ptr
#endif

#if USE_ALGOR!=4 || SELECT_BLAS!=2
    init_random<Float>(&AA[0], num_row_m*n_size);
#else
    init_random<Float>(&AA_device[0], num_row_m*n_size);
#endif

#if USE_ALGOR==1 || USE_ALGOR==2 || USE_ALGOR==3
#if USE_OMP>0
    #pragma omp parallel for
#endif
    for(int i=0 ; i<num_row_m*p_size ; ++i){
      CC[i] = 0. ;
    }
#endif

    // --- --- ---

    {
#if USE_MPI>1
      MPI_Waitall(mpi_split_num_call, &BB_request[0], MPI_STATUS_IGNORE);
#if USE_ALGOR==4 && SELECT_BLAS==2 && (!defined(USE_MPI_GPU_DIRECT) || USE_MPI_GPU_DIRECT==0)
      libsci_acc_Memcpy(&BB_device[0], &BB[0],
                        n_size*p_size * sizeof(Float), libsci_acc_MemcpyHTD);
#endif
#endif
    }

    // --- --- ---

#if defined(CRAYPAT) && USE_PAT_API>1
    PAT_region_begin(2,"matmul_xxx");
#endif

#if USE_ALGOR!=1 && USE_ALGOR!=2 && USE_ALGOR!=3 && USE_ALGOR!=4
    matmul_trivial<Float>(CC, AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==1
    matmul_loop_interchange<Float>(CC, AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==2
    matmul_cache_blocking<Float>(CC, AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==3
    matmul_micro_kernel<Float>(CC, AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==4
#if SELECT_BLAS!=2
    matmul_cblas<Float>(CC, AA, BB, 
		        num_row_m, p_size, n_size,
			p_size, n_size, p_size
		       );
#else // SELECT_BLAS
  #if USE_OMP==0
    matmul_libsci_acc<Float>(CC, AA, BB, 
		             CC_device, AA_device, BB_device,
		             num_row_m, p_size, n_size
			    );
  #else
    std::cerr << "ERROR :: Using libsci_acc.h with OpenMP is NOT supported!" << std::endl;
  #endif
#endif // SELECT_BLAS
#endif // USE_ALGOR

#if defined(CRAYPAT) && USE_PAT_API>1
    PAT_region_end(2);
#endif

    // --- --- ---

#if WRITE_ARRAYS>0
    {
      if( niter % int(NUM_ITER_PER_WRITE) == 0 )
      {
#if USE_ALGOR==4 && SELECT_BLAS==2
	libsci_acc_Memcpy(AA, AA_device, num_row_m*n_size * sizeof(Float), libsci_acc_MemcpyDTH);
        libsci_acc_Memcpy(CC, CC_device, num_row_m*p_size * sizeof(Float), libsci_acc_MemcpyDTH);
#endif
#if USE_MPI>0 && USE_MPI_IO<1
	MPI_Request gather_req[2] ;
        if( mpi_rank == 0 ){
          MPI_Igatherv(MPI_IN_PLACE, mpi_aa_recv_size[0], CUSTOM_MPI_FLOAT,
                      &AA[0], mpi_aa_recv_size.data(), mpi_aa_recv_disp.data(), CUSTOM_MPI_FLOAT,
                      0, MPI_COMM_WORLD, &gather_req[0]);
	  MPI_Igatherv(MPI_IN_PLACE, mpi_cc_recv_size[0], CUSTOM_MPI_FLOAT,
                      &CC[0], mpi_cc_recv_size.data(), mpi_cc_recv_disp.data(), CUSTOM_MPI_FLOAT,
                      0, MPI_COMM_WORLD, &gather_req[1]);
        }else{
          MPI_Igatherv(&AA[0], num_row_m*n_size, CUSTOM_MPI_FLOAT,
                      nullptr, nullptr, nullptr, CUSTOM_MPI_FLOAT,
                      0, MPI_COMM_WORLD, &gather_req[0]);
	  MPI_Igatherv(&CC[0], num_row_m*p_size, CUSTOM_MPI_FLOAT,
                      nullptr, nullptr, nullptr, CUSTOM_MPI_FLOAT,
                      0, MPI_COMM_WORLD, &gather_req[1]);
        }
	MPI_Waitall(2, &gather_req[0], MPI_STATUS_IGNORE);
#endif
    
    // --- --- ---

#if USE_MPI_IO>0
#if USE_ALGOR==4 && SELECT_BLAS==2 && USE_MPI_GPU_DIRECT>0
        libsci_acc_Memcpy(&BB[start_row_n*p_size], &BB_device[start_row_n*p_size], num_row_n*p_size * sizeof(Float), libsci_acc_MemcpyDTH);
#endif
        write_array_mpi<Float,CUSTOM_MPI_FLOAT>(
                        &AA[0], start_row_m*n_size, num_row_m*n_size,
                        "A_"+std::to_string(niter)+".bin"
	  	        );
        if( mpi_rank == 0 )
          std::cout << "  A matrix is written in binary format using MPI-IO." << std::endl;
     
        write_array_mpi<Float,CUSTOM_MPI_FLOAT>(
                        &BB[start_row_n*p_size], start_row_n*p_size, num_row_n*p_size,
                        "B_"+std::to_string(niter)+".bin"
                        );
        if( mpi_rank == 0 )
          std::cout << "  B matrix is written in binary format using MPI-IO." << std::endl;

        write_array_mpi<Float,CUSTOM_MPI_FLOAT>(
                        &CC[0], start_row_m*p_size, num_row_m*p_size,
                        "C_"+std::to_string(niter)+".bin"
                        );
        if( mpi_rank == 0 )
          std::cout << "  C matrix is written in binary format using MPI-IO." << std::endl;

#else
        if( mpi_rank == 0 )
        {
#if USE_ALGOR==4 && SELECT_BLAS==2 && USE_MPI_GPU_DIRECT>0
          libsci_acc_Memcpy(BB, BB_device, n_size*p_size * sizeof(Float), libsci_acc_MemcpyDTH);
#endif
          write_array<Float>(m_size, n_size, AA, "A_"+std::to_string(niter));
          std::cout << "  A matrix is written." << std::endl;

          write_array<Float>(n_size, p_size, BB, "B_"+std::to_string(niter));
          std::cout << "  B matrix is written." << std::endl;

          write_array<Float>(m_size, p_size, CC, "C_"+std::to_string(niter));
          std::cout << "  C matrix is written." << std::endl;
        }
#endif // USE_MPI_IO
      } // niter % NUM_ITER_PER_WRITE
    } 
#endif // WRITE_ARRAYS

  } // Main iteration loop ==  NUM_REPEAT

#if USE_MPI>0
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#if defined(CRAYPAT) && USE_PAT_API>0
  PAT_region_end(1);
#endif
  stop_clock(mpi_rank, clock_type::loop);

  // ------------------------------

#if USE_ALGOR==4 && SELECT_BLAS==2
  libsci_acc_DeviceFree(AA_device);
  libsci_acc_DeviceFree(BB_device);
  libsci_acc_DeviceFree(CC_device);
  AA_device = BB_device = CC_device = nullptr ;
#endif

  std::free(AA);
  std::free(BB);
  std::free(CC);
  AA = BB = CC = nullptr ;

  // ------------------------------

  if( mpi_rank == 0 ){
    std::cout << "\n-----" << std::endl;
  }
  print_clock(mpi_rank, clock_type::loop, "Main loops.");
  stop_clock(mpi_rank, clock_type::prog);
  print_clock(mpi_rank, clock_type::prog, "Whole prog.");

  // ------------------------------


#if USE_ALGOR==4 && SELECT_BLAS==2
  libsci_acc_finalize();
#endif

#if USE_MPI>0
  MPI_Finalize();
#endif

return 0; }



