import numpy as np
def R2(y_data,y_model):
    return 1- np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

#returns a regular (exponent increases by one) design matrix from data x and polynomial of degree n
def DesignMatrix(x,polydgree):
    X = np.zeros((len(x),polydgree+1))
    for i in range(polydgree+1):
        X[:,i] = x**i
    return X

#Implements Ridge Regression using Design matrix (X_training) training data of y (y_training),
#  the lambda coefficient, and the degree of the approximating polynomial.
#Returns the beta coeffs. and their variance
def RidgeRegression(X_training,y_training,Lambda,PolyApproxDegree):
    I = np.eye(PolyApproxDegree)
    inverse_matrix = np.linalg.inv(X_training.T @ X_training+Lambda*I)
    beta_variance = np.diagonal(inverse_matrix)
    beta = inverse_matrix @ X_training.T @ y_training
    return beta, beta_variance

#Implements Ridge Regression using Design matrix (X_training) training data of y (y_training)
#Returns the beta coeffs. and their variance
def LinearRegression(X_training,y_training):
    inverse_matrix = np.linalg.inv(X_training.T @ X_training)
    beta_variance = np.diagonal(inverse_matrix)
    beta = inverse_matrix @ X_training.T @ y_training
    return beta, beta_variance

#Returns the evaluation of a polynomial with the coefficients beta at point x
def Coeff_to_Poly(x,beta):
    poly = 0
    for i in range(len(beta)):
        poly += beta[i]*x**i
    return poly

#The Franke Function.
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Design Matrix for a 2D-polynomial.

def DesignMatrix_deg2(x,y,polydegree,include_intercept=False):
    """
    Input: x,y (created as meshgrids), the maximal polynomial degree, if the intercept should be included or not.

    The function creates a design matrix X for a 2D-polynomial.

    Returns: The design matrix X
    """
    adder=0 #The matrix dimension is increased by one if include_intercept is True
    p=round((polydegree+1)*(polydegree+2)/2)-1 #The total amount of coefficients
    if include_intercept:
        p+=1
        adder=1
    datapoints=len(x)
    X=np.zeros((datapoints,p))
    if include_intercept:
        X[:,0]=1
    X[:,0+adder]=x # Adds x on the first column
    X[:,1+adder]=y # Adds y on the second column
    """
    Create the design matrix:
    X[:,3] is x**2 * y**0, X[:,4] is x**1 * y **1,
    X[:,7] is x**2 * y ** 1 and so on. Up to degree degree
    """
    count=2+adder
    for i in range(2,polydegree+1):
        for j in range(i+1):
            X[:,count]=X[:,0+adder]**j*X[:,1+adder]**(i-j)
            count+=1;
    return X
