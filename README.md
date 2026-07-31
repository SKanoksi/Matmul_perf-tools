# Matmul_perf-tools
Matrix multiplication program for teaching performance profiling and optimization using HPE Cray Perftools (CrayPAT)

Note: Not take account of edge case <-> Matrix dimension length must be divisible by block size.

#define USE_ALGOR 0
//  0 = Trivial loops (default)
//  1 = Loop interchange
//  2 = Loop interchange + Cache blocking (no packing)
//  3 = Micro kernel + Cache blocking (packing)
//  4 = CBLAS (cblas_dgemm)

#define SELECT_BLAS 0
//  0 = cblas.h
//  1 = mkl.h
//  2 = libsci_acc.h (GPU)

#define USE_OMP 0
//  0 = No OpenMP
//  1 = Simple, omp parallel for
//  2 = Simple, omp parallel for collapse(..)

#define USE_MPI 0
//  0 = No MPI
//  1 = Blocking, MPI_Bcast
//  2 = Non-blocking, MPI_Ibcast
//  3 = Non-blocking, MPI_Igatherv

#define USE_MPI_GPU_DIRECT 0
//  0 = Standard MPI
//  1 = with GPUDirect RDMA

#define WRITE_ARRAYS 0
#define NUM_ITER_PER_WRITE 10
#define USE_MPI_IO 0
//  0 = Gather -> Serial  
//  1 = Non-collective
//  2 = Collective

