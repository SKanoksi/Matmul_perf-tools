/******************************************************************\

  Matmul -- perf tools

  Version 1.0.0
  Copyright (c) 2026, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
\******************************************************************/

#ifndef MATMUL_UTIL_HPP
#define MATMUL_UTIL_HPP

#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>

#include "matmul_setup.hpp"


inline void partition_dim(std::size_t &start_x, std::size_t &local_size_x, std::size_t &stripe_size_x, 
     	                  const std::size_t id, const std::size_t group_size, const std::size_t x_size)
{
  std::size_t temp_size_x = x_size/group_size ;
  temp_size_x = (group_size*temp_size_x < x_size) ? temp_size_x+1 : temp_size_x ;

  stripe_size_x = temp_size_x ;
  start_x = id*stripe_size_x ;

  temp_size_x = (id+1) * stripe_size_x ;
  temp_size_x = (temp_size_x<x_size) ? temp_size_x : x_size ; // end_x
  local_size_x = temp_size_x - start_x ;

return; }


#if USE_OMP>0
static thread_local std::mt19937_64 gen;
#else
static std::mt19937_64 gen;
#endif

void init_random_gen()
{
#if USE_OMP>0
  #pragma omp parallel
  {
    std::random_device rd ;
    gen.seed(rd() ^ (uint64_t)omp_get_thread_num());
  }
#else
  std::random_device rd;
  gen.seed(rd());
#endif
}

inline uint32_t fast_hash(uint32_t x) 
{
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

template <typename U>
void init_random(U *__restrict A, std::size_t array_size, const U min_rand=-1.0, const U max_rand=1.0)
{
#if USE_OMP==1
  #pragma omp parallel
  {
#endif
    const uint32_t seed = static_cast<uint32_t>(gen()) ;
    
#if USE_OMP>0
    #pragma omp for
#endif
    for(std::size_t i = 0; i<array_size; ++i)
    {
      A[i] = min_rand + (max_rand - min_rand) * (static_cast<U>(fast_hash(seed + (uint32_t)i))/static_cast<U>(UINT32_MAX)) ;
    }
#if USE_OMP==1
  }
#endif

return; }


template <typename U>
bool write_array_bin(const U *A, std::size_t size,
                     const std::string &filename)
{
  std::ofstream ofile(filename, std::ios::out | std::ios::binary);
  
  if( ofile.is_open() ){
    ofile.write(reinterpret_cast<const char*>(A), size*sizeof(U));
    ofile.close();
  }else{
    std::cerr
    << "Error cannot open file, " << filename
    << ", for writing output." << std::endl;
    return false;
  }

return true; }


template <typename U>
bool write_array_txt(const std::size_t num_row, const std::size_t num_col, 
                     const U *A,
                     const std::string &filename, const char *delim)
{
  std::ofstream ofile(filename, std::ios::out);

  if( !ofile.is_open() ){
    std::cerr 
    << "Error cannot open file, " << filename  
    << ", for writing output." << std::endl;
    return false; 
  }

  ofile << std::fixed << std::setprecision(15);

  for(std::size_t r=0 ; r < num_row ; ++r)
  {
    const std::size_t start = r * num_col ;
    for(std::size_t c=0 ; c < num_col ; ++c)
    {
      ofile << A[start+c] << delim ;
    }
    ofile << std::endl;
  }

  ofile.close();
  std::cout << "  Data is written to " << filename << std::endl;

return true; }


template <typename U>
bool write_array(const std::size_t num_row, const std::size_t num_col,
                 const U *A,
                 const std::string &filename)
{
#if USE_TXT_FORMAT>0
  return write_array_txt<U>(num_row, num_col, A, filename+".txt", " ");
#else
  return write_array_bin<U>(A, num_row*num_col, filename+".bin");
#endif
}


enum clock_type { all=0, computation=1 } ;

#if USE_TIMER>0

std::chrono::time_point<std::chrono::high_resolution_clock>  start[2], end[2] ;
std::chrono::duration<double>  total_duration[2] ;

inline void start_clock(const int mpi_rank, clock_type type){
  if( mpi_rank == 0 ){
    start[static_cast<int>(type)] = std::chrono::high_resolution_clock::now();
  }
}
inline void stop_clock(const int mpi_rank, clock_type type){
  if( mpi_rank == 0 ){
    const int i = static_cast<int>(type) ;
    end[i] = std::chrono::high_resolution_clock::now();
    total_duration[i] += std::chrono::duration<double>(end[i] - start[i]) ;
  }
}
inline void reset_clock(const int mpi_rank, clock_type type){
  if( mpi_rank == 0 ){
    total_duration[static_cast<int>(type)] = std::chrono::duration<double>::zero();
  }
}
inline void print_clock(const int mpi_rank, clock_type type, const char *section_name){
  if( mpi_rank == 0 ){
    std::cout
    << "Time :: " << section_name << " = "
    << total_duration[static_cast<int>(type)].count() 
    << " seconds." << std::endl;
  }
}

#else

inline void start_clock(const int mpi_rank, clock_type type){ return; }
inline void stop_clock(const int mpi_rank, clock_type type){ return; }
inline void reset_clock(const int mpi_rank, clock_type type){ return; }
inline void print_clock(const int mpi_rank, clock_type type, const char *section_name){ return; }

#endif

#endif  // MATMUL_UTIL_HPP

