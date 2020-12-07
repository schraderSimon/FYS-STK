import scipy.io as sio
import numpy as np
import sys
from function_library import *
def test_matrix_algo():
    A=np.array([[1,2,3],[4,5,6],[7,8,9]])
    tridiagonal_indices=np.tril_indices(len(A[0]))
    sorted=np.argsort(np.linalg.norm(A,axis=1)) #order of column norms
    print(sorted)
    Anew=A[:,np.flip(sorted)] # sort the coulomb matrix
    print(A)
    print(Anew)
    print(A[tridiagonal_indices]) # Only take the tridiagonal indeces
def sort_coulomb_matrix(A):
    Anew=np.zeros_like(A)
    sorted=np.flip(np.argsort(np.linalg.norm(A,axis=1))) #order of column norms
    for i in range(len(sorted)):
        for j in range(len(sorted)):
            Anew[i][j]=A[sorted[i],sorted[j]]
    return Anew
print(sort_coulomb_matrix(A))


def
