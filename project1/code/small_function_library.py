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