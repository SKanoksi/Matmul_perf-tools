# Matmul_perf-tools
Matrix multiplication program for teaching performance optimization using HPE Cray Perftools (CrayPAT)

-DUSE_ALGOR=
//  0 = Trivial loops (default)
//  1 = Loop interchange
//  2 = Loop interchange + Cache blocking (no packing)
//  3 = Micro kernel + Cache blocking (packing)
//  4 = CBLAS (cblas_dgemm)

-DSELECT_BLAS=
// For USE_ALGOR=4
//  0 = cblas.h
//  1 = mkl.h
//  2 = libsci_acc.h

-DUSE_OMP=
//  0 = No OpenMP
//  1 = Simple, omp parallel for
//  2 = Separate, omp parallel + omp for

-DUSE_MPI=
//  0 = No OpenMP
//  1 = Blocking
//  2 = Non-blocking

-DUSE_MPI_IO=
//  0 = Gather -> Serial
//  1 = Non-collective
//  2 = Collective
