import numpy as np
import sys

iter_range = np.arange(0, int(sys.argv[1]), int(sys.argv[5]))
float_datatype = np.float64

isBinary = True
M = int(sys.argv[2])
N = int(sys.argv[3])
P = int(sys.argv[4])

Verbose = False

for n in iter_range:
    if not isBinary :
        # For standard text file
        A = np.loadtxt("A_"+str(n)+".txt", dtype=float_datatype)
        B = np.loadtxt("B_"+str(n)+".txt", dtype=float_datatype)
        C = np.loadtxt("C_"+str(n)+".txt", dtype=float_datatype)
    else:    
        # For binary when using MPI_IO
        A = np.fromfile("A_"+str(n)+".bin", dtype=float_datatype).reshape([M,N])
        B = np.fromfile("B_"+str(n)+".bin", dtype=float_datatype).reshape([N,P])
        C = np.fromfile("C_"+str(n)+".bin", dtype=float_datatype).reshape([M,P])

    Cpy = np.matmul(A, B)

    if Verbose :
        print("A =", A)
        print("B =", B)
        print("C =", C)
        print("Cpy =", Cpy)

    rela_devia = np.abs(np.divide(Cpy - C, C)).flatten()
    print("--- ITER =",n,"---")
    print("NumPy allclose() =", np.allclose(Cpy, C))
    print("Average relative error is around", np.average(rela_devia)*100, "percents.")
    print("Maximum relative error is around", np.max(rela_devia)*100, "percents.\n")


