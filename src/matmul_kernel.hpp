/******************************************************************\

  Matmul -- perf tools

  Version 1.0.0
  Copyright (c) 2026, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
\******************************************************************/

#ifndef MATMUL_KERNEL_HPP
#define MATMUL_KERNEL_HPP

#include "matmul_setup.hpp"


template <typename U, int num_rows, int num_cols, int num_col_packs>
void pack_array(U *__restrict__ dst, const U *__restrict__ src,
                const int src_row_factor, const int src_col_factor, const int src_pack_factor)
{
  for(int p=0 ; p<num_col_packs ; ++p)
  {
    for(int r=0 ; r<num_rows ; ++r)
    {
      for(int c=0 ; c<num_cols ; ++c)
      {
        dst[num_rows*num_cols*p+num_cols*r+c] = src[p*src_pack_factor + r*src_row_factor + c*src_col_factor] ;
      }
    }
  }

return; }


// {M,P} = {N,M} x {N,P}
// {i,j} = {k,i} x {k,j}
template <typename U, int Mc, int Pc, int num_k>
void micro_kernel_scalar(U *__restrict C, const int num_col_C, const U *__restrict A, const U *__restrict B)
{
  alignas(SIMD_ALIGNED_BYTE) U c[Mc*Pc] = {0} ;

  // Adding factors along N (index k)
  for(int k=0 ; k < num_k ; ++k)
  {
    // Outer product
    for(int i=0 ; i < Mc ; ++i)
    {
      for(int j=0 ; j < Pc ; ++j)
        c[Pc*i+j] += A[Mc*k+i]*B[Pc*k+j];
    }
  }

  // Write back
  for(int i=0 ; i < Mc ; ++i)
  {
    for(int j=0 ; j < Pc ; ++j)
      C[num_col_C*i+j] += c[Pc*i+j] ;
  }

return; }


#if (defined(__AVX__) || defined(__AVX2__)) && (!defined(ALGOR3_NO_MANUAL_VEC_KERNEL) || ALGOR3_NO_MANUAL_VEC_KERNEL<1)
template <int Mc, int Pc_div_4, int num_k>
void micro_kernel_avx2_pd(double *__restrict C, const int num_col_C, const double *__restrict A, const double *__restrict B)
{
  typedef double v4df __attribute__((vector_size(32))) ;

  v4df c[Mc*Pc_div_4] = {0.0} ;

  for(int k=0 ; k < num_k ; ++k)
  {
    const int ia = Mc*k ;
    const int jb = Pc_div_4*4*k ;
    for(int i=0 ; i < Mc ; ++i)
    {
      for(int j=0 ; j < Pc_div_4 ; ++j)
      {
        v4df b_vec = {B[jb+4*j+0], B[jb+4*j+1], B[jb+4*j+2], B[jb+4*j+3]} ;
        c[Pc_div_4*i+j] += A[ia+i] * b_vec ;
      }
    }
  } // k

  for(int i=0 ; i < Mc ; ++i)
  {
    for(int j=0 ; j < Pc_div_4 ; ++j)
    {
      C[num_col_C*i+4*j+0] += c[Pc_div_4*i+j][0] ;
      C[num_col_C*i+4*j+1] += c[Pc_div_4*i+j][1] ;
      C[num_col_C*i+4*j+2] += c[Pc_div_4*i+j][2] ;
      C[num_col_C*i+4*j+3] += c[Pc_div_4*i+j][3] ;
    }
  }

return; }


template <int Mc, int Pc_div_8, int num_k>
void micro_kernel_avx2_ps(float *__restrict C, const int num_col_C, const float *__restrict A, const float *__restrict B)
{
  typedef float v8sf __attribute__((vector_size(32))) ;

  v8sf c[Mc*Pc_div_8] = {0.0f} ;

  for(int k=0 ; k < num_k ; ++k)
  {
    const int ia = Mc*k ;
    const int jb = Pc_div_8*8*k ;
    for(int i=0 ; i < Mc ; ++i)
    {
      for(int j=0 ; j < Pc_div_8 ; ++j)
      {
        v8sf b_vec = {B[jb+8*j+0], B[jb+8*j+1], B[jb+8*j+2], B[jb+8*j+3], 
                      B[jb+8*j+4], B[jb+8*j+5], B[jb+8*j+6], B[jb+8*j+7]} ;
        c[Pc_div_8*i+j] += A[ia+i] * b_vec ;
      }
    }
  } // k

  for(int i=0 ; i < Mc ; ++i)
  {
    for(int j=0 ; j < Pc_div_8 ; ++j)
    {
      C[num_col_C*i+8*j+0] += c[Pc_div_8*i+j][0] ;
      C[num_col_C*i+8*j+1] += c[Pc_div_8*i+j][1] ;
      C[num_col_C*i+8*j+2] += c[Pc_div_8*i+j][2] ;
      C[num_col_C*i+8*j+3] += c[Pc_div_8*i+j][3] ;
      C[num_col_C*i+8*j+4] += c[Pc_div_8*i+j][4] ;
      C[num_col_C*i+8*j+5] += c[Pc_div_8*i+j][5] ;
      C[num_col_C*i+8*j+6] += c[Pc_div_8*i+j][6] ;
      C[num_col_C*i+8*j+7] += c[Pc_div_8*i+j][7] ;
    }
  }

return; }
#endif

template <typename U, int Mc, int Pc, int num_k>
void micro_kernel(U *__restrict C, const int num_col_C, const U *__restrict A, const U *__restrict B)
{
#if (defined(__AVX__) || defined(__AVX2__)) && (!defined(ALGOR3_NO_MANUAL_VEC_KERNEL) || ALGOR3_NO_MANUAL_VEC_KERNEL<1)
  if constexpr ( sizeof(U) == 8 && Pc % 4 == 0 ){
    micro_kernel_avx2_pd<Mc,Pc/4,num_k>(C, num_col_C, A, B);
  }else if constexpr ( sizeof(U) == 4 && Pc % 8 == 0 ){
    micro_kernel_avx2_ps<Mc,Pc/8,num_k>(C, num_col_C, A, B);
  }else{
    micro_kernel_scalar<U,Mc,Pc,num_k>(C, num_col_C, A, B);
  }
#else
  micro_kernel_scalar<U,Mc,Pc,num_k>(C, num_col_C, A, B);
#endif

return; }


#endif  // MATMUL_KERNEL_HPP

