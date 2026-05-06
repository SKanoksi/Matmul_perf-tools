import numpy as np

iter_range = np.arange(0,100,10)
float_datatype = np.float64

isBinary = True
M = 3000
N = 3000
P = 3000

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

    rela_devia = np.abs(np.divide(Cpy - C, C)).flatten()
    print("--- ITER =",n,"---")
    print("NumPy allclose() =", np.allclose(Cpy, C))
    print("Average relative error is around", np.average(rela_devia)*100, "percents.")
    print("Maximum relative error is around", np.max(rela_devia)*100, "percents.\n")
