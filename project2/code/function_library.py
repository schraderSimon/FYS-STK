import numpy as np
import sys
from numba import jit, int32, float32
from numba.experimental import jitclass
def DesignMatrix(x,polydgree):
    """
    Input: Data x (array), the desired maximum degree (int)
    Output: The reular design matrix (exponent increases by one) for a polynomial of degree polydgree
    """
    X = np.zeros((len(x),polydgree+1))
    for i in range(polydgree+1):
        X[:,i] = x**i
    return X

def R2(y_data,y_model):
    """
    Input: The original target data, the fitted data
    Output: The R2 value.
    """
    return 1- np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def MSE(y_data,y_model):
    """
    Input: Two arrays of equal length
    Output: The MSE between the two vectors
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def RidgeRegression(X_training,y_training,Lambda, include_beta_variance=True):

    """
    Input: The design matrix X, and the targets Y and a value for LAMBDA, and whether the variance should be included too.
    Output: The Ridge Regression beta and the variance (zero if include_beta_variance is zero)
    This was implemented as SVD.

    """
    I = np.eye(len(X_training[0,:]))
    if include_beta_variance:
        inverse_matrix = np.linalg.inv(X_training.T @ X_training+Lambda*I)
        beta_variance = np.diagonal(inverse_matrix)
    else:
        beta_variance=0
    u, s, vh = np.linalg.svd(X_training, full_matrices=False)
    smat=np.zeros((vh.shape[0],u.shape[1]))
    for i in range(len(s)):
        smat[i][i]=s[i]
    beta= vh.T @ (np.linalg.inv(smat.T@smat+(I*Lambda)) @ smat.T) @ u.T @ y_training
    return beta, beta_variance
#Implements Ridge Regression using Design matrix (X_training) training data of y (y_training)
#Returns the beta coeffs. and their variance

def LinearRegression(X_training,y_training,include_beta_variance=True):
    """Input: The design matrix X, and the targets Y
        Output: The OLS beta and the variance (zero if include_beta_variance is zero)
    """
    if include_beta_variance:
        inverse_matrix = np.linalg.inv(X_training.T @ X_training)
        beta_variance = np.diag(inverse_matrix)
    else:
        beta_variance=0
    u, s, vh = np.linalg.svd(X_training, full_matrices=False)
    beta= vh.T @ np.linalg.inv(np.diag(s)) @ u.T @ y_training
    return beta, beta_variance
class SGD:
    def __init__(self,X,y,n_epochs=100,theta=0):
        self.n_epochs=n_epochs;
        self.X=X;
        self.y=y;
        self.n=len(X) #Number of rows
        self.p=len(X[0]) #number of columns

        if theta==0:
            self.theta=np.random.randn(self.p,1) #Create start guess for theta if theta is not given as parameter
        else:
            self.theta=theta
    def learning_schedule(self,t,t0,t1):
        return t0/(t+t1)
    def calculateGradient(self,theta, index=-1):
        if index == -1:
            index=np.random.randint(self.n) #If the index is not explicitely set, choose a random index
        xi= self.X[index:index+1]
        yi= self.y[index:index+1]
        gradients = 2 * xi.T @ ((xi @ theta)-yi)
        return gradients
    def simple_fit(self,eta=0.01):
        theta=self.theta
        for epoch in range(1,self.n_epochs+1): #For each epoch
            for i in range(self.n): #For each point
                theta=theta-eta*self.calculateGradient(theta);
        return theta.ravel()
    def decay_fit(self,t0=5,t1=50):
        theta=self.theta
        for epoch in range(1,self.n_epochs+1):
            for i in range(self.n):
                gradient=self.calculateGradient(theta);
                eta=self.learning_schedule(epoch+self.n*i,t0,t1)
                theta=theta-eta*gradient
        return theta.ravel()
    def RMSprop(self,eta=1e-2,cbeta=0.9, error=1e-8):
        theta=self.theta
        s=np.zeros_like(theta)
        for epoch in range(1,self.n_epochs+1):
            for i in range(self.n):
                gradient=self.calculateGradient(theta)
                s=cbeta*s+(1-cbeta)*(gradient*gradient)
                theta= theta-eta*gradient/np.sqrt(s+error)
        return theta.ravel()
