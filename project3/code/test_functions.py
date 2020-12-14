import scipy.io as sio
import numpy as np
import sys
from function_library import *
from numpy import sqrt
def test_matrix_algo():
    "Test that the algorithms for the coulomb matrix work properly"
    "Example Molecule: H 2 0 0, O 0 1 0, H 0 0 1, C 0 0 0"
    A=np.array([[0.5,1/sqrt(5),3,8/sqrt(5)],[1/sqrt(5),0.5,6,8/sqrt(2)],
                [3,6,6**2.4/2,48],[8/sqrt(5),8/sqrt(2),48,8**2.4/2]])
    A_sorted_exp=np.array([[8**2.4/2,48,8/sqrt(2),8/sqrt(5)],[48,6**2.4/2,6,3],
                [8/sqrt(2),6,1/2,1/sqrt(5)],[8/sqrt(5),3,1/sqrt(5),1/2]])
    A_sorted_calc=sort_coulomb_matrix(A)
    for i in range(len(A)):
        for j in range(len(A)):
            """test that the sort algorithm is correct"""
            assert abs(A_sorted_exp[i,j]-A_sorted_calc[i,j])<1e-8
    A_reduced_exp=[[8**2.4/2,48,6**2.4/2,8/sqrt(2),6,1/2,8/sqrt(5),3,1/sqrt(5),1/2]]
    A_reduced_calc=reduce_coulomb([A])
    for i in range(len(A_reduced_exp[0])):
        """test that the reducing algorithm is correct"""
        assert abs(A_reduced_exp[0][i]-A_reduced_calc[0][i])<1e-8
    A_noH_exp=np.array([[[6**2.4/2,6*8],[6*8,8**2.4/2],]])
    A_noH_calc=create_hydrogenfree_coulomb_matrix([A],4)
    for i in range(2):
        for j in range(2):
            "Test that the reduced coulomb matrix is correct"
            assert abs((A_noH_calc[0,i,j]-A_noH_exp[0,i,j]))<1e-8
test_matrix_algo()
