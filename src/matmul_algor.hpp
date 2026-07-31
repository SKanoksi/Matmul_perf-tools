/******************************************************************\

  Matmul -- Perf. tools

  Version 2.0.0
  Copyright (c) 2026, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.

\******************************************************************/

#ifndef MATMUL_ALGOR_HPP
#define MATMUL_ALGOR_HPP

#if USE_ALGOR==4
#if SELECT_BLAS==1
  #include <mkl.h>
#elif SELECT_BLAS==2
  #include <libsci_acc.h>
#else
  #include <cblas.h>
#endif
#endif

#include <matmul_setup.hpp>

#if USE_ALGOR==3
  #include <matmul_kernel.hpp>
#endif

template <typename U>
void matmul_trivial(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                    const int m_size, const int p_size, const int n_size)
{
#if USE_OMP>0
#if USE_OMP>1
  #pragma omp parallel for collapse(2) schedule(runtime)
#else
  #pragma omp parallel for schedule(runtime)
#endif
#endif
  for(int i=0 ; i < m_size ; ++i)
  for(int j=0 ; j < p_size ; ++j)
  {
    U temp = 0. ;
    for(int k=0 ; k < n_size ; ++k)
    {
      temp += AA[n_size*i+k] * BB[p_size*k+j] ;
    }
    CC[p_size*i+j] = temp ;
  }

return; }


#if USE_ALGOR==1
template <typename U>
void matmul_loop_interchange(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                             const int m_size, const int p_size, const int n_size)
{
#if USE_OMP>0
#if USE_OMP>1
  #warning "Using USE_ALGOR=1 with USE_OMP=2 requires atomic orcritical"
  #pragma omp parallel for collapse(2) schedule(runtime)
#else
  #pragma omp parallel for schedule(runtime)
#endif
#endif
  for(int i=0 ; i < m_size ; ++i)
  for(int k=0 ; k < n_size ; ++k)
  {
    for(int j=0 ; j < p_size ; ++j)
    {
#if USE_OMP>1
      Float value = AA[n_size*i+k] * BB[p_size*k+j];
      //#pragma omp atomic/critical
      #pragma omp atomic
      CC[p_size*i+j] += value;
#else
      CC[p_size*i+j] += AA[n_size*i+k] * BB[p_size*k+j] ;
#endif
    }
  }

return; }
#endif


#if USE_ALGOR==2
template <typename U>
void matmul_cache_blocking(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                           const int m_size, const int p_size, const int n_size)
{

#if USE_OMP>0
#if USE_OMP>1
  #pragma omp parallel for collapse(2) schedule(runtime)
#else
  #pragma omp parallel for schedule(runtime)
#endif
#endif
  for(int jj=0 ; jj < p_size ; jj+=BLOCK_P_SIZE)
  for(int ii=0 ; ii < m_size ; ii+=BLOCK_M_SIZE)
  for(int kk=0 ; kk < n_size ; kk+=BLOCK_N_SIZE)
  {
    for(int i=ii ; i < ii+BLOCK_M_SIZE ; ++i)
    for(int k=kk ; k < kk+BLOCK_N_SIZE ; ++k)
    {
      const U AA_value = AA[n_size*i+k] ;
      for(int j=jj ; j < jj+BLOCK_P_SIZE ; ++j)
      {
        CC[p_size*i+j] += AA_value * BB[p_size*k+j] ;
      }
    }
  }

return; }
#endif


#if USE_ALGOR==3
template <typename U>
void matmul_micro_kernel(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                         const int m_size, const int p_size, const int n_size)
{
  alignas(SIMD_ALIGNED_BYTE) static U BB_block[BLOCK_N_SIZE*MICRO_P_SIZE*NUM_PANEL_P] ;

#if !defined(USE_OMP) || USE_OMP<2

#if !defined(ALGOR3_MERGE_LOOP_PACK_LOCAL) || ALGOR3_MERGE_LOOP_PACK_LOCAL<1

  alignas(SIMD_ALIGNED_BYTE) static U AA_block_trans[BLOCK_N_SIZE*MICRO_M_SIZE*NUM_PANEL_M] ;
#if USE_OMP>0
  #pragma omp parallel for private(BB_block,AA_block_trans) schedule(runtime)
#endif
  for(int jj=0 ; jj < p_size ; jj+=BLOCK_P_SIZE)
  {
    for(int kk=0 ; kk < n_size ; kk+=BLOCK_N_SIZE)
    {
      pack_array<U,BLOCK_N_SIZE,MICRO_P_SIZE,NUM_PANEL_P>(BB_block,&BB[p_size*kk+jj], p_size,1,MICRO_P_SIZE);

      for(int ii=0 ; ii < m_size ; ii+=BLOCK_M_SIZE)
      {
        pack_array<U,BLOCK_N_SIZE,MICRO_M_SIZE,NUM_PANEL_M>(AA_block_trans,&AA[n_size*ii+kk], 1,n_size,MICRO_M_SIZE*n_size);

        // ----- From cache blocking to micro kernel -----
        for(int ic=0 ; ic < BLOCK_M_SIZE ; ic+=MICRO_M_SIZE)
        {
          for(int jc=0 ; jc < BLOCK_P_SIZE ; jc+=MICRO_P_SIZE)
          {
            micro_kernel<U,MICRO_M_SIZE,MICRO_P_SIZE,BLOCK_N_SIZE>
                        (&CC[p_size*(ii+ic)+(jj+jc)], p_size,
                         &AA_block_trans[BLOCK_N_SIZE*ic],
                         &BB_block[BLOCK_N_SIZE*jc]
                        );
          } // jc
        } // ic
        // ----- From micro kernel to cache blocking -----
      } // ii
    } // kk
  } //jj
    
#else // ALGOR3_MERGE_LOOP_PACK_LOCAL

  alignas(SIMD_ALIGNED_BYTE) static U AA_block_trans[BLOCK_N_SIZE*MICRO_M_SIZE] ;
#if USE_OMP>0
  #pragma omp parallel for private(BB_block,AA_block_trans) schedule(runtime)
#endif
  for(int jj=0 ; jj < p_size ; jj+=BLOCK_P_SIZE)
  {
    for(int kk=0 ; kk < n_size ; kk+=BLOCK_N_SIZE)
    {
      pack_array<U,BLOCK_N_SIZE,MICRO_P_SIZE,NUM_PANEL_P>
                (BB_block,&BB[p_size*kk+jj], p_size,1,MICRO_P_SIZE);

      for(int ii=0 ; ii < m_size ; ii+=MICRO_M_SIZE)
      {
        pack_array<U,BLOCK_N_SIZE,MICRO_M_SIZE,1>
                  (AA_block_trans,&AA[n_size*ii+kk], 1,n_size, 0);

        for(int jc=0 ; jc < BLOCK_P_SIZE ; jc+=MICRO_P_SIZE)
        {
          micro_kernel<U,MICRO_M_SIZE,MICRO_P_SIZE,BLOCK_N_SIZE>
                      (&CC[p_size*ii+(jj+jc)], p_size,
                       AA_block_trans,
                       &BB_block[BLOCK_N_SIZE*jc]
                      );
        } // jc
      } // ii
    } // kk
  } //jj

#endif // ALGOR3_MERGE_LOOP_PACK_LOCAL


#else
  alignas(SIMD_ALIGNED_BYTE) static U AA_block_trans[BLOCK_N_SIZE*MICRO_M_SIZE*NUM_PANEL_M] ;
  //alignas(SIMD_ALIGNED_BYTE) static U AA_block_trans[BLOCK_N_SIZE*MICRO_M_SIZE] ;
  
  #pragma omp parallel for collapse(2) private(BB_block,AA_block_trans) schedule(runtime)
  for(int jj=0 ; jj < p_size ; jj+=BLOCK_P_SIZE)
  for(int ii=0 ; ii < m_size ; ii+=BLOCK_M_SIZE)
  for(int kk=0 ; kk < n_size ; kk+=BLOCK_N_SIZE)
  {
    pack_array<U,BLOCK_N_SIZE,MICRO_P_SIZE,NUM_PANEL_P>(BB_block,&BB[p_size*kk+jj], p_size,1,MICRO_P_SIZE);
    pack_array<U,BLOCK_N_SIZE,MICRO_M_SIZE,NUM_PANEL_M>(AA_block_trans,&AA[n_size*ii+kk], 1,n_size,MICRO_M_SIZE*n_size);

    // ----- From cache blocking to micro kernel -----
    for(int ic=0 ; ic < BLOCK_M_SIZE ; ic+=MICRO_M_SIZE)
    {
      //pack_array<U,BLOCK_N_SIZE,MICRO_M_SIZE,1>(AA_block_trans,&AA[n_size*(ii+ic)+kk], 1,n_size,0);

      for(int jc=0 ; jc < BLOCK_P_SIZE ; jc+=MICRO_P_SIZE)
      {
        micro_kernel<U,MICRO_M_SIZE,MICRO_P_SIZE,BLOCK_N_SIZE>
                    (&CC[p_size*(ii+ic)+(jj+jc)], p_size,
                     &AA_block_trans[BLOCK_N_SIZE*ic],
                     &BB_block[BLOCK_N_SIZE*jc]
                    );
      } // jc
    } // ic
    // ----- From micro kernel to cache blocking -----
  }
#endif 

return; }
#endif


#if USE_ALGOR==4 
#if SELECT_BLAS!=2
template <typename U>
void matmul_cblas(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                  const int m_size, const int p_size, const int n_size,
		  const int ldc, const int lda, const int ldb)
{
  if constexpr ( sizeof(U) == 8 ){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m_size, p_size, n_size,
                1.0,
                reinterpret_cast<const double*>(AA), lda,
                reinterpret_cast<const double*>(BB), ldb,
                0.0,
                reinterpret_cast<double*>(CC), ldc);
  }else if constexpr ( sizeof(U) == 4 ){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m_size, p_size, n_size,
                1.0f,
                reinterpret_cast<const float*>(AA), lda,
                reinterpret_cast<const float*>(BB), ldb,
                0.0f,
                reinterpret_cast<float*>(CC), ldc);
  }else{
    static_assert(sizeof(U)==4 || sizeof(U)==8, "matmul_cblas only supports double and float");
  }

return; }

#else // SELECT_CBLAS

template <typename U>
void matmul_libsci_acc(U *__restrict CC, U *__restrict AA, U *__restrict BB,
		       U *CC_device, U *AA_device, U *BB_device,
                       const int m_size, const int p_size, const int n_size)
{
  if constexpr ( sizeof(U) == 8 ){
    dgemm_acc('N', 'N', p_size, m_size, n_size,
              1.0,
              reinterpret_cast<double*>(BB_device), p_size,
              reinterpret_cast<double*>(AA_device), n_size,
              0.0,
              reinterpret_cast<double*>(CC_device), p_size);
  }else if constexpr ( sizeof(U) == 4 ){
    sgemm_acc('N', 'N', p_size, m_size, n_size,
              1.0f,
              reinterpret_cast<float*>(BB_device), p_size,
              reinterpret_cast<float*>(AA_device), n_size,
              0.0f,
              reinterpret_cast<float*>(CC_device), p_size);
  }else{
    static_assert(sizeof(U)==4 || sizeof(U)==8, "matmul_libsci_acc only supports double and float");
  }

return; }

#endif // SELECT_BLAS
#endif // USE_ALGOR==4

#endif // MATMUL_ALGOR_HPP

