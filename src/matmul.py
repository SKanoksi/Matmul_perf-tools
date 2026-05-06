import numpy as np
import sys

REPEAT = 100

M_SIZE = 3000
N_SIZE = 3000
P_SIZE = 3000

if len(sys.argv) > 1 :
    REPEAT = int(sys.argv[1])
if len(sys.argv) > 2 :
    M_SIZE = int(sys.argv[2])
if len(sys.argv) > 3 :
    N_SIZE = int(sys.argv[3])
if len(sys.argv) > 4 :
    P_SIZE = int(sys.argv[4])

float_datatype = np.float64

for n in range(REPEAT):
    print("Begin iteration",n)

    A = np.random.uniform(-1.0, 1.0, size=(M_SIZE,N_SIZE)).astype(float_datatype)
    B = np.random.uniform(-1.0, 1.0, size=(N_SIZE,P_SIZE)).astype(float_datatype)
    C = np.matmul(A, B)


