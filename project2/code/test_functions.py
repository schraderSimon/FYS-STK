import numpy as np
from function_library import *
from sklearn.linear_model import SGDRegressor, Ridge
def test_optimizers():
    """
    This function tests wether all implemented Gradient Descent methods
    give the same result as the expected analytical result within a given tolerance
    (we choose 1e--1 as tolerance, given that these methods are not accurate)
    We test for batchsize 1 and batchsize 5
    """
    x=np.random.randn(100)
    y=5*x+3
    X=DesignMatrix(x,2) # create a design matrix
    theta=np.array([3,5,0]) #analytical results
    tol=1e-1
    for batchsize in [1, 5]:
        sgd=SGD(X,y,1000,batchsize=batchsize)
        theta_RMSprop=sgd.RMSprop(eta=0.005)
        theta_normal=sgd.simple_fit(eta=0.01) #simple SGD
        theta_decay=sgd.decay_fit(t0=50,t1=5000) #SGD
        error=[sum(abs(theta_RMSprop-theta)),sum(abs(theta_normal-theta))]
        error.append(sum(abs(theta_decay-theta)))
        assert error[0] <tol, "%.3f"%error[0]
        assert error[1] < tol, "%.3f"%error[1]
        assert error[2] < tol, "%.3f"%error[2]
def test_optimizers_ridge():
    x=np.random.randn(100)
    y=5*x+3
    X=DesignMatrix(x,2) # create a design matrix
    tol=1e-1
    Lambda=0.1
    theta, unimportant=RidgeRegression(X,y,Lambda,False)
    for batchsize in [32,16]: #This fails HORRIBLY for 1
        sgd=SGD_Ridge(X,y,1000,batchsize=batchsize,Lambda=Lambda)
        theta_RMSprop=sgd.RMSprop(eta=0.005)
        theta_normal=sgd.simple_fit(eta=0.01) #simple SGD
        theta_decay=sgd.decay_fit(t0=50,t1=5000) #SGD
        error=[sum(abs(theta_RMSprop-theta)),sum(abs(theta_normal-theta))]
        error.append(sum(abs(theta_decay-theta)))
        assert error[0] <tol, "%.3f"%error[0]
        assert error[1] < tol, "%.3f"%error[1]
        assert error[2] < tol, "%.3f"%error[2]

test_optimizers_ridge()
test_optimizers()
prediction_train=nn.predict(X_train_scaled)
prediction_test=nn.predict(X_test_scaled)
print("Test MSE: %f" %MSE(prediction_test,z_test_scaled))
print("Train MSE: %f" %MSE(prediction_train,z_train_scaled))
print(prediction_train[:5])
print(z_train_scaled[:5])
