/******************************************************************\

  Matmul -- perf tools

  Version 1.0.0
  Copyright (c) 2026, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
\******************************************************************/

#include "matmul_setup.hpp"

#if USE_MPI>0
  #include <mpi.h>
#endif

#if USE_OMP>0
  #include <omp.h>
#endif

#if defined(CRAYPAT) && USE_PAT_API>0
  #include <pat_api.h>
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
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_provided);
  if( mpi_provided < MPI_THREAD_SERIALIZED )
  {
    std::cerr << "The MPI_THREAD_SERIALIZED is NOT supported." << std::endl;
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
    std::cout << "Using " << mpi_size << " MPI processes" << std::endl;
#else
  {
    std::cout << "Using 1 single process without MPI" << std::endl;
#endif
#if USE_OMP>0
  #pragma omp parallel
  {
    if( omp_get_thread_num()==0 ){
      num_threads = omp_get_num_threads();
      std::cout << "Using " << num_threads << " OMP threads" << std::endl;
    }
  }
#else
  std::cout << "Using 1 single thread without OpenMP" << std::endl;
#endif
  }

#if USE_ALGOR==4 && SELECT_BLAS==2
  libsci_acc_init();
  if( mpi_rank == 0 )
    std::cout << "Using GPU threads" << std::endl;
#endif

  // ---

  std::size_t start_row_m, num_row_m, stripe_size_m ;
  partition_dim(start_row_m, num_row_m, stripe_size_m, 
                mpi_rank, mpi_size, m_size);

#if WRITE_ARRAYS>0 && USE_MPI_IO>0
  std::size_t start_row_n, num_row_n, stripe_size_n ;
  partition_dim(start_row_n, num_row_n, stripe_size_n,
                mpi_rank, mpi_size, n_size);
#endif

  // ---
  
#if USE_ALGOR==2 || USE_ALGOR==3
  if( stripe_size_m % BLOCK_M_SIZE != 0 )
  {
    if( mpi_rank == 0 ){
      std::cerr
      << "\nINPUT_ERROR :: "
      << "stripe_size_m, i.e., ceil(matrix_m_size/mpi_size), must be divisible by BLOCK_M_SIZE.\n"
      << std::endl;
    }
#if USE_MPI>0
    //MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
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
    //MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
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
    //MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    return 1;
  }
#endif


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
#if USE_OMP>0
#if USE_ALGOR!=2 && USE_ALGOR!=3 && USE_ALGOR!=4
    std::cout
    << "Working matrix size: "
    << "[" << stripe_size_m/num_threads << "," << p_size << "] = ["
    << stripe_size_m/num_threads << "," << n_size << "] x ["
    << n_size << "," << p_size << "]"
    << " <-- OpenMP"
    << std::endl;
#elif USE_ALGOR==2 || USE_ALGOR==3 || (USE_ALGOR==4 && USE_OMP==2)
    std::cout
    << "Working matrix size: "
    << "[" << stripe_size_m << "," << p_size/num_threads << "] = ["
    << stripe_size_m << "," << n_size << "] x ["
    << n_size << "," << p_size/num_threads << "]"
    << " <-- OpenMP"
    << std::endl;
#endif
#endif
#if USE_ALGOR==2 || USE_ALGOR==3
    std::cout
    << "  Matrix block size: "
    << "[" << BLOCK_M_SIZE << "," << BLOCK_P_SIZE << "] = ["
    << BLOCK_M_SIZE << "," << BLOCK_N_SIZE << "] x ["
    << BLOCK_N_SIZE << "," << BLOCK_P_SIZE << "]"
    << " <-- Cache blocking"
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
#if defined(ALGOR3_NO_MERGE_LOOP) && ALGOR3_NO_MERGE_LOOP>0
    << " -- NO MERGE LOOP"
#endif
#if defined(ALGOR3_NO_MANUAL_VEC_KERNEL) && ALGOR3_NO_MANUAL_VEC_KERNEL>0
    << " -- NO VEC KERNEL"
#endif

#elif USE_ALGOR==4
#if SELECT_BLAS==1
    << "[4] = BLAS lvl3 from <mkl.h>"
#elif SELECT_BLAS==2
    << "[4] = BLAS lvl3 from <libsci_acc.h>" 
#else
    << "[4] = BLAS lvl3 from <cblas.h>"
#endif
#else
    << "[0] = Trivial loops"
#endif
    << "\n" << std::endl;

    reset_clock(mpi_rank, clock_type::all);
    reset_clock(mpi_rank, clock_type::computation);
  }

  // ------------------------------

  const int alloc_size_m = (mpi_rank==0) ? m_size : stripe_size_m ;
  Float *__restrict AA = (Float*)std::aligned_alloc(SIMD_ALIGNED_BYTE,
                                                    alloc_size_m*n_size * sizeof(Float));
  Float *__restrict BB = (Float*)std::aligned_alloc(SIMD_ALIGNED_BYTE,
                                                    n_size*p_size * sizeof(Float));
  Float *__restrict CC = (Float*)std::aligned_alloc(SIMD_ALIGNED_BYTE,
                                                    (stripe_size_m*mpi_size)*p_size * sizeof(Float));

#if USE_MPI>0
  if( (stripe_size_m*mpi_size)*p_size * sizeof(Float) > 1024*1024*1024 && mpi_rank == 0 )
  {
    std::cout << "\n Warning :: your C matrix may be too large for MPI_Allgather\n" << std::endl;
  }
  if( n_size*p_size * sizeof(Float) > 1024*1024*1024 && mpi_rank == 0 )
  {
    std::cout << "\n Warning :: your B matrix may be too large for MPI_Bcast\n" << std::endl;
  }
#endif

#if USE_ALGOR==4 && SELECT_BLAS==2
  Float *AA_device, *BB_device, *CC_device ;
  libsci_acc_DeviceAlloc((void **)&AA_device, num_row_m*n_size * sizeof(Float));
  libsci_acc_DeviceAlloc((void **)&BB_device,    n_size*p_size * sizeof(Float));
  libsci_acc_DeviceAlloc((void **)&CC_device, num_row_m*p_size * sizeof(Float));
#endif

  init_random_gen();

#if USE_MPI>0
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // ------------------------------

  start_clock(mpi_rank, clock_type::all);

#if defined(CRAYPAT) && USE_PAT_API>0
  PAT_region_begin(1,"iteration_loop");
#endif

#if USE_MPI>1
  MPI_Request bcast_BB_request, allgather_CC_request ;
#endif

#if USE_OMP>1
  #pragma omp parallel
  {
#if USE_ALGOR==4 && USE_OMP>1
    std::size_t start_col_p, num_col_p, stripe_size_p ;
    partition_dim(start_col_p, num_col_p, stripe_size_p,
                  omp_get_thread_num(), omp_get_num_threads(), p_size);
#endif
#endif
  for(int niter=0 ; niter < num_repeat ; ++niter)
  {
    if( mpi_rank == 0 ){  
#if PRINT_LOOP_ITER>0
#if USE_OMP>1
      #pragma omp single nowait
#endif
      std::cout << "Begin loop " << niter+1 << "/" << num_repeat << std::endl;
#endif
      init_random<Float>(BB, n_size*p_size);
      // Implicit barrier
    }

#if USE_OMP>1 && USE_MPI>0
    #pragma omp single nowait
#endif
    {
#if USE_MPI==1
      MPI_Bcast(&BB[0], n_size*p_size, CUSTOM_MPI_FLOAT, 0, MPI_COMM_WORLD);
#elif USE_MPI>1
      MPI_Ibcast(&BB[0], n_size*p_size, CUSTOM_MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD, &bcast_BB_request);
#endif 
    }

    init_random<Float>(&AA[0], stripe_size_m*n_size);
    // Implicit barrier
    //   Require one between MPI_Ibcast and MPI_Wait 
    //   since MPI_THREAD_SERIALIZED is used

#if USE_MPI>1 && USE_MPI_IO>0
    if( niter!=0 ){
#if USE_OMP>1
      #pragma omp single
#endif
      {
        MPI_Wait(&allgather_CC_request, MPI_STATUS_IGNORE);
      } // <-- Implicit barrier of omp single 
    }
#endif

#if USE_ALGOR==1 || USE_ALGOR==2 || USE_ALGOR==3
#if USE_OMP>1
    #pragma omp for nowait
#elif USE_OMP==1
    #pragma omp parallel for
#endif
    for(int i=start_row_m ; i<(start_row_m+stripe_size_m)*p_size ; ++i){
      CC[i] = 0. ;
    }
#endif

    // --- --- ---

#if USE_OMP>1
    #pragma omp single
#endif
    {
#if USE_MPI>1
      MPI_Wait(&bcast_BB_request, MPI_STATUS_IGNORE);
#endif
      start_clock(mpi_rank, clock_type::computation);
    }
    // <-- Implicit barrir of omp single


    // --- --- ---

#if defined(CRAYPAT) && USE_PAT_API>1
    PAT_region_begin(2,"matmul_xxx");
#endif

#if USE_ALGOR!=1 && USE_ALGOR!=2 && USE_ALGOR!=3 && USE_ALGOR!=4
    matmul_trivial<Float>(&CC[p_size*start_row_m], AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==1
    matmul_loop_interchange<Float>(&CC[p_size*start_row_m], AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==2
    matmul_cache_blocking<Float>(&CC[p_size*start_row_m], AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==3
    matmul_micro_kernel<Float>(&CC[p_size*start_row_m], AA, BB, num_row_m, p_size, n_size);
#elif USE_ALGOR==4
#if SELECT_BLAS!=2
  #if USE_OMP>1
    matmul_cblas<Float>(&CC[p_size*start_row_m+start_col_p], AA, &BB[start_col_p], 
                        num_row_m, num_col_p, n_size,
                        p_size, n_size, p_size
                        );
    #pragma omp barrier
  #else
    matmul_cblas<Float>(&CC[p_size*start_row_m], AA, BB, 
                        num_row_m, p_size, n_size,
                        p_size, n_size, p_size
                        );
  #endif
#else // SELECT_BLAS
  #if USE_OMP==0
    matmul_libsci_acc<Float>(&CC[p_size*start_row_m], AA, BB, 
                             CC_device, AA_device, BB_device,
                             num_row_m, p_size, n_size);
  #else
    std::cerr << "ERROR :: Using libsci_acc.h with OpenMP is NOT supported!" << std::endl;
  #endif
#endif // SELECT_BLAS
#endif // USE_ALGOR
  // <-- Implicit barrir of omp for
  //     or explict barrier (cblas)

#if defined(CRAYPAT) && USE_PAT_API>1
    PAT_region_end(2);
#endif

    // --- --- ---

#if USE_OMP>1 && (USE_MPI>0 || USE_TIMER>0)
    #pragma omp single 
#endif
    {
      stop_clock(mpi_rank, clock_type::computation);
#if USE_MPI>0
#if USE_MPI>1
      MPI_Iallgather(MPI_IN_PLACE, stripe_size_m*p_size, CUSTOM_MPI_FLOAT,
                     &CC[0], stripe_size_m*p_size, CUSTOM_MPI_FLOAT,
                     MPI_COMM_WORLD, &allgather_CC_request);
#else
      MPI_Allgather(MPI_IN_PLACE, stripe_size_m*p_size, CUSTOM_MPI_FLOAT,
                    &CC[0], stripe_size_m*p_size, CUSTOM_MPI_FLOAT,
                    MPI_COMM_WORLD);
#endif
#if WRITE_ARRAYS>0 && USE_MPI_IO<1
      if( niter % int(NUM_ITER_PER_WRITE) == 0 )
      {
        if( mpi_rank == 0 ){
          MPI_Gather(MPI_IN_PLACE, stripe_size_m*n_size, CUSTOM_MPI_FLOAT,
                    &AA[0], stripe_size_m*n_size, CUSTOM_MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        }else{
          MPI_Gather(&AA[0], stripe_size_m*n_size, CUSTOM_MPI_FLOAT,
                    nullptr, stripe_size_m*n_size, CUSTOM_MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        }
      }
#endif
#endif
    }
    // <-- Implicit barrier of omp single
    //     required between MPI_Iallgather (n) and MPI_Ibcast (n+1)
    //     and obviously before wring AA and CC

    // --- --- ---

#if WRITE_ARRAYS>0
#if USE_OMP>1
    #pragma omp single
#endif
    {
      if( niter % int(NUM_ITER_PER_WRITE) == 0 )
      {
#if USE_MPI_IO>0
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
                        &CC[start_row_m*p_size], start_row_m*p_size, num_row_m*p_size,
                        "C_"+std::to_string(niter)+".bin"
                        );
        if( mpi_rank == 0 )
          std::cout << "  C matrix is written in binary format using MPI-IO." << std::endl;

#else
        if( mpi_rank == 0 )
        {
          write_array<Float>(m_size, n_size, AA, "A_"+std::to_string(niter));
          std::cout << "  A matrix is written." << std::endl;

          write_array<Float>(n_size, p_size, BB, "B_"+std::to_string(niter));
          std::cout << "  B matrix is written." << std::endl;

#if USE_MPI>1
          MPI_Wait(&allgather_CC_request, MPI_STATUS_IGNORE);
#endif
          write_array<Float>(m_size, p_size, CC, "C_"+std::to_string(niter));
          std::cout << "  C matrix is written." << std::endl;
#if USE_MPI>1
        }else{
          MPI_Wait(&allgather_CC_request, MPI_STATUS_IGNORE);
        }
#else
        }
#endif

#endif // USE_MPI_IO
      } // niter % NUM_ITER_PER_WRITE
    } // Implicit barrier of omp single
#endif // WRITE_ARRAYS

  } // *** NUM_REPEAT loop ***

#if USE_OMP>1
  } // #pragma omp parallel
#endif


#if USE_MPI>1
  // Last call
  if( num_repeat>0 )
  {
    MPI_Wait(&allgather_CC_request, MPI_STATUS_IGNORE);
  }
#endif
#if defined(CRAYPAT) && USE_PAT_API>0
  PAT_region_end(1);
#endif
  stop_clock(mpi_rank, clock_type::all);

  // ------------------------------

  if( mpi_rank == 0 ){
    std::cout << "\n-----" << std::endl;  
  }
  print_clock(mpi_rank, clock_type::computation,  "Main comp.");
  print_clock(mpi_rank, clock_type::all,          "Main loops");

  // ------------------------------

#if USE_ALGOR==4 && SELECT_BLAS==2
  libsci_acc_DeviceFree(AA_device);
  libsci_acc_DeviceFree(BB_device);
  libsci_acc_DeviceFree(CC_device);
  AA_device = BB_device = CC_device = nullptr ;
#endif

  free(AA);
  free(BB);
  free(CC);
  AA = BB = CC = nullptr ;

#if USE_ALGOR==4 && SELECT_BLAS==2
  libsci_acc_finalize();
#endif

#if USE_MPI>0
  MPI_Finalize();
#endif

return 0; }



