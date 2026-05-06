/******************************************************************\

  Matmul -- perf tools

  Version 1.0.0
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

#include "matmul_setup.hpp"
#include "matmul_kernel.hpp"


template <typename U>
void matmul_trivial(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                    const int m_size, const int p_size, const int n_size)
{
#if USE_OMP>1
  #pragma omp for schedule(runtime)
#elif USE_OMP==1
  #pragma omp parallel for schedule(runtime)
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


template <typename U>
void matmul_loop_interchange(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                             const int m_size, const int p_size, const int n_size)
{
#if USE_OMP>1
  #pragma omp for schedule(runtime)
#elif USE_OMP==1
  #pragma omp parallel for schedule(runtime)
#endif
  for(int i=0 ; i < m_size ; ++i)
  for(int k=0 ; k < n_size ; ++k)
  {
    for(int j=0 ; j < p_size ; ++j)
    {
      CC[p_size*i+j] += AA[n_size*i+k] * BB[p_size*k+j] ;
    }
  }

return; }


template <typename U>
void matmul_cache_blocking(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                           const int m_size, const int p_size, const int n_size)
{
#if USE_OMP>1
  #pragma omp for schedule(runtime)
#elif USE_OMP==1
  #pragma omp parallel for schedule(runtime)
#endif
  //  {M,P}  =  {M,N}  x  {N,P}
  // {ii,jj} = {ii,kk} x {kk,jj}
  for(int jj=0 ; jj < p_size ; jj+=BLOCK_P_SIZE)
  {
    for(int kk=0 ; kk < n_size ; kk+=BLOCK_N_SIZE)
    {
      for(int ii=0 ; ii < m_size ; ii+=BLOCK_M_SIZE)
      {
        for(int i=ii ; i < ii+BLOCK_M_SIZE ; ++i)
        {
          for(int k=kk ; k < kk+BLOCK_N_SIZE ; ++k)
          {
            const U AA_value = AA[n_size*i+k] ;
            for(int j=jj ; j < jj+BLOCK_P_SIZE ; ++j)
            {
              CC[p_size*i+j] += AA_value * BB[p_size*k+j] ;
            }
          }
        }
      }
    }
  }
return; }


template <typename U>
void matmul_micro_kernel(U *__restrict CC, const U *__restrict AA, U *__restrict BB,
                         const int m_size, const int p_size, const int n_size)
{
  alignas(SIMD_ALIGNED_BYTE) static U BB_block[BLOCK_N_SIZE*MICRO_P_SIZE*NUM_PANEL_P] ;
#if !defined(ALGOR3_NO_MERGE_LOOP) || ALGOR3_NO_MERGE_LOOP<1
  alignas(SIMD_ALIGNED_BYTE) static U AA_block_trans[BLOCK_N_SIZE*MICRO_M_SIZE] ;
#if USE_OMP>1
  #pragma omp for private(BB_block,AA_block_trans) schedule(runtime)
#elif USE_OMP==1
  #pragma omp parallel for private(BB_block,AA_block_trans) schedule(runtime)
#endif
  //  {M,P}  =  {M,N}  x  {N,P}
  // {ii,jj} = {ii,kk} x {kk,jj}
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
#else
  alignas(SIMD_ALIGNED_BYTE) static U AA_block_trans[BLOCK_N_SIZE*MICRO_M_SIZE*NUM_PANEL_M] ;
#if USE_OMP>1
  #pragma omp for private(BB_block,AA_block_trans) schedule(runtime)
#elif USE_OMP==1
  #pragma omp parallel for private(BB_block,AA_block_trans) schedule(runtime)
#endif
  //  {M,P}  =  {M,N}  x  {N,P}
  // {ii,jj} = {ii,kk} x {kk,jj}
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
#endif

return; }


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
  libsci_acc_Memcpy(AA_device, AA, m_size*n_size * sizeof(U), libsci_acc_MemcpyHTD);
  libsci_acc_Memcpy(BB_device, BB, n_size*p_size * sizeof(U), libsci_acc_MemcpyHTD);

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
  libsci_acc_Memcpy(CC, CC_device, m_size*p_size * sizeof(U), libsci_acc_MemcpyDTH);

return; }

#endif // SELECT_BLAS
#endif // USE_ALGOR==4

#endif // MATMUL_ALGOR_HPP

