import numpy as np
from small_function_library import *
tol=1e-12
def test_matrix_creation():
    """Test whether the Design Matrix is set up properly"""
    testmatrix=np.zeros((3,6))
    x=np.array([0,1,2])
    y=np.array([1,2,3])
    testmatrix[:,0]=1;
    testmatrix[:,1]=y;
    testmatrix[:,2]=x;
    testmatrix[:,3]=x**2;
    testmatrix[:,4]=x*y;
    testmatrix[:,5]=y**2;
    calc_matrix=DesignMatrix_deg2(y,x,2,include_intercept=True)
    for i in range(3):
        for j in range(6):
            assert abs(calc_matrix[i,j]-testmatrix[i,j])<tol
def test_linear_regression():
    """Test wether OLS is correctly implemented using a simple straight function"""
    test_alpha=0.3
    test_beta=0.7
    x=np.linspace(-5,5,100)
    y=test_alpha*x+test_beta
    X=DesignMatrix(x,2) # create a fit where we expect the coefficient for X**2 to be zero
    beta,betavar=LinearRegression(X,y)
    assert abs(beta[0]-test_beta)<tol
    assert abs(beta[1]-test_alpha)<tol
    assert abs(beta[2])<tol

def test_ridge():
    """Test wether Ridge is correctly implemented using a simple straight function at lambda=0"""
    test_alpha=0.3
    test_beta=0.7
    x=np.linspace(-5,5,100)
    y=test_alpha*x+test_beta
    X=DesignMatrix(x,2)
    beta,betavar=RidgeRegression(X,y,0)
    assert abs(beta[0]-test_beta)<tol
    assert abs(beta[1]-test_alpha)<tol
    assert abs(beta[2])<tol
def test_lasso():
    """Test wether Lasso is correctly implemented using a simple straight function at lambda=0 (even though it doesn't work well)"""
    test_alpha=0.3
    test_beta=0.7
    x=np.linspace(-5,5,100)
    y=test_alpha*x+test_beta
    X=DesignMatrix(x,2)
    beta=LASSORegression(X,y,0,tol=tol,iter=1e5)
    assert abs(beta[0]-test_beta)<tol
    assert abs(beta[1]-test_alpha)<tol
    assert abs(beta[2])<tol
def test_R2():
    """Test wether the R2 score of to identical arrays is equal to one"""
    y=np.random.randn(1000)
    ystrek=np.copy(y)
    assert abs(R2(ystrek,y)-1)<tol
def test_MSE():
    """Test wether the MSE sore of to identical arrays is equal to zero"""
    y=np.random.randn(1000)
    ystrek=np.copy(y)
    assert abs(MSE(ystrek,y))<tol
test_matrix_creation()
test_linear_regression()
test_ridge()
test_lasso()
test_R2()
test_MSE()

"""
run as: python3 testfunksjoner.py
"""
