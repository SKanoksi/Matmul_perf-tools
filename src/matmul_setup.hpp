/******************************************************************\

  Matmul -- perf tools

  Version 1.0.0
  Copyright (c) 2026, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
\******************************************************************/

#ifndef MATMUL_SETUP_HPP
#define MATMUL_SETUP_HPP

// *** General ***
constexpr int  DEFAULT_NUM_REPEAT = 100 ; // arg0

// *** Matrix size ***
// {M,P} = {M,N} x {N,P}
constexpr int  DEFAULT_MATRIX_M_SIZE = 3000 ; // arg1
constexpr int  DEFAULT_MATRIX_N_SIZE = 3000 ; // arg2
constexpr int  DEFAULT_MATRIX_P_SIZE = 3000 ; // arg3

// *** Block size (for ALGOR=2,3) *** 
constexpr int  BLOCK_M_SIZE = 1000 ; 
constexpr int  BLOCK_N_SIZE = 1000 ;
constexpr int  BLOCK_P_SIZE = 1000 ; 

// *** Micro size (for ALGOR=3) ***
constexpr int  MICRO_M_SIZE = 2 ;
constexpr int  MICRO_P_SIZE = 2 ;


// --------- DEFAULT --------
// Passing -DXXX=N at compile-time
//  takes precedent over below defaults

// *** Algorithm (default) ***
#if !defined(USE_ALGOR)
  #define USE_ALGOR 0
//  0 = Trivial loops (default)
//  1 = Loop interchange
//  2 = Loop interchange + Cache blocking (no packing)
//  3 = Micro kernel + Cache blocking (packing)
//  4 = CBLAS (cblas_dgemm)
#endif
#if !defined(SELECT_BLAS)
  #define SELECT_BLAS 0
//  0 = cblas.h
//  1 = mkl.h
//  2 = libsci_acc.h 
#endif


// *** Floating-point datatype ***
// Choose  1) double, MPI_DOUBLE
//   or    2) float,  MPI_FLOAT
using Float = double ;
#define CUSTOM_MPI_FLOAT MPI_DOUBLE


// *** Parallel (default) ***
#if !defined(USE_MPI)
  #define USE_MPI 0
//  0 = No OpenMP
//  1 = Blocking
//  2 = Non-blocking
#endif
#if !defined(USE_OMP)
  #define USE_OMP 0
//  0 = No OpenMP
//  1 = Simple, omp parallel for
//  2 = Separate, omp parallel + omp for
#endif
#if USE_ALGOR==4 && SELECT_BLAS==2 && USE_OMP!=0
  #error "Using libsci_acc.h with OpenMP is NOT supported."
#endif


// *** IO write (default) ***
#if !defined(WRITE_ARRAYS)
  #define WRITE_ARRAYS 0
#endif
#if !defined(NUM_ITER_PER_WRITE) || NUM_ITER_PER_WRITE<1
  #define NUM_ITER_PER_WRITE 10
#endif
#if !defined(USE_TXT_FORMAT)
  #define USE_TXT_FORMAT 0
#endif
#if !defined(USE_MPI_IO)
  #if USE_MPI==1
    #define USE_MPI_IO 1  // Non-collective
  #elif USE_MPI>1
    #define USE_MPI_IO 2  // Collective
  #else
    #define USE_MPI_IO 0  // Gather -> Serial
  #endif
#endif
#if USE_MPI_IO>0 && USE_MPI<1
  #error "USE_MPI_IO>0 needs USE_MPI>0"
#endif
#if USE_MPI_IO>0 && USE_TXT_FORMAT!=0
  #error "USE_MPI_IO>0 does NOT support writing in text format"
#endif


// *** Other (default) ***
#if !defined(PRINT_LOOP_ITER)
  #define PRINT_LOOP_ITER 1
#endif
#if !defined(USE_PAT_API)
  #define USE_PAT_API 0
#endif
#if !defined(USE_TIMER)
  #define USE_TIMER 1
#endif
#if !defined(USE_PRAGMA) // Enable PRAGMAs below if used
  #define USE_PRAGMA 1
#endif


// --------------------------------------

static_assert(BLOCK_M_SIZE % MICRO_M_SIZE == 0,
              "BLOCK_M_SIZE must be divisible by MICRO_M_SIZE.");
static_assert(BLOCK_P_SIZE % MICRO_P_SIZE == 0,
              "BLOCK_P_SIZE must be divisible by MICRO_P_SIZE.");
constexpr int NUM_PANEL_M = BLOCK_M_SIZE/MICRO_M_SIZE ;
constexpr int NUM_PANEL_P = BLOCK_P_SIZE/MICRO_P_SIZE ;


#if defined(__AVX512F__)
  #define SIMD_ALIGNED_BYTE 64
#elif defined(__AVX__) || defined(__AVX2__)
  #define SIMD_ALIGNED_BYTE 32
#else
  #define SIMD_ALIGNED_BYTE 16
#endif
constexpr int SIMD_NUM_E = SIMD_ALIGNED_BYTE/sizeof(Float) ;


#if USE_PRAGMA>0
#define STRINGIFY(x)        #x
#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#define PRAGMA_VECTORIZE    _Pragma("ivdep")
#define PRAGMA_NO_VECTOR    _Pragma("novector")
#define PRAGMA_UNROLL_AUTO  _Pragma("unroll")
#define PRAGMA_NO_UNROLL    _Pragma("nounroll")
#define PRAGMA_UNROLL(n)    _Pragma(STRINGIFY(unroll(n)))
#elif defined(__clang__)
#define PRAGMA_VECTORIZE    _Pragma("clang loop vectorize(enable)")
#define PRAGMA_NO_VECTOR    _Pragma("clang loop vectorize(disable)")
#define PRAGMA_UNROLL_AUTO  _Pragma("clang loop unroll(enable)")
#define PRAGMA_NO_UNROLL    _Pragma("clang loop unroll(disable)")
#define PRAGMA_UNROLL(n)    _Pragma(STRINGIFY(clang loop unroll_count(n)))
#elif defined(__GNUC__)
#define PRAGMA_VECTORIZE    _Pragma("GCC ivdep")
#define PRAGMA_NO_VECTOR    _Pragma("GCC novector")
#define PRAGMA_UNROLL_AUTO  
#define PRAGMA_NO_UNROLL    _Pragma("GCC unroll 0")
#define PRAGMA_UNROLL(n)    _Pragma(STRINGIFY(GCC unroll n))
#endif
#endif

#if !defined(PRAGMA_VECTORIZE)
#define PRAGMA_VECTORIZE
#endif
#if !defined(PRAGMA_NO_VECTOR)
#define PRAGMA_NO_VECTOR
#endif
#if !defined(PRAGMA_UNROLL_AUTO)
#define PRAGMA_UNROLL_AUTO
#endif
#if !defined(PRAGMA_NO_UNROLL)
#define PRAGMA_NO_UNROLL
#endif
#if !defined(PRAGMA_UNROLL)
#define PRAGMA_UNROLL(n)
#endif


#endif // MATMUL_SETUP_HPP

