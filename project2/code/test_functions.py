import numpy as np
from function_library import *
from sklearn.linear_model import Ridge
def test_optimizers():
    """
    Tests Stochastic Gradient Descent methods using OLS Regression.

    This function tests wether all implemented Gradient Descent methods
    give the same result as the expected analytical result within a given tolerance
    (we choose 1e--1 as tolerance, given that these methods are not accurate)
    We test for batchsizes 5 and batchsize 1
    """
    x=np.random.randn(100)
    y=5*x+3
    X=DesignMatrix(x,2) # create a design matrix
    theta=np.array([3,5,0]) #analytical results
    tol=1e-1
    for batchsize in [1, 5]:
        sgd=SGD(X,y,1000,batchsize=batchsize)
        theta_RMSprop=sgd.RMSprop(eta=0.005); sgd.reset()
        theta_normal=sgd.simple_fit(eta=0.01); sgd.reset() #simple SGD
        theta_decay=sgd.decay_fit(t0=50,t1=5000); sgd.reset() #SGD
        theta_adam=sgd.ADAM(eta=0.01); sgd.reset()
        error=[sum(abs(theta_RMSprop-theta)),sum(abs(theta_normal-theta))]
        error.append(sum(abs(theta_decay-theta)))
        error.append(sum(abs(theta_adam-theta)))
        assert error[0] <tol, "%.3f"%error[0]
        assert error[1] < tol, "%.3f"%error[1]
        assert error[2] < tol, "%.3f"%error[2]
        assert error[3] < tol, "%.3f"%error[3]
def test_optimizers_ridge():
    """
    Tests Stochastic Gradient Descent methods using Ridge Regression.

    This function tests wether all implemented Gradient Descent methods
    give the same result as the expected analytical result within a given tolerance
    (we choose 1e--1 as tolerance, given that these methods are not accurate)
    We test for batchsizes 32 and batchsize 16
    """

    x=np.random.randn(100)
    y=5*x+3
    X=DesignMatrix(x,2) # create a design matrix
    tol=1e-1
    Lambda=0.1
    theta, unimportant=RidgeRegression(X,y,Lambda,False)
    for batchsize in [32,16]: #This fails HORRIBLY for 1
        sgd=SGD_Ridge(X,y,1000,batchsize=batchsize,Lambda=Lambda); sgd.reset()
        theta_RMSprop=sgd.RMSprop(eta=0.005); sgd.reset()
        theta_normal=sgd.simple_fit(eta=0.01); sgd.reset() #simple SGD
        theta_decay=sgd.decay_fit(t0=50,t1=5000); sgd.reset() #SGD decay
        theta_adam=sgd.ADAM(eta=0.01); sgd.reset()
        error=[sum(abs(theta_RMSprop-theta)),sum(abs(theta_normal-theta))]
        error.append(sum(abs(theta_decay-theta)))
        error.append(sum(abs(theta_adam-theta)))
        assert error[0] <tol, "%.3f"%error[0]
        assert error[1] < tol, "%.3f"%error[1]
        assert error[2] < tol, "%.3f"%error[2]
        assert error[3] < tol, "%.3f"%error[3]
def test_NN_reg():
    """Tests Neural Network for a regression problem."""
    x=np.random.randn(100)
    y=5*x
    X=DesignMatrix(x,2) # create a design matrix
    tol=1e-1
    Lambda=0.1
    hidden_neurons=[100,50,30,20]
    nn=NeuralNetwork(X,y,
        n_hidden_layers=len(hidden_neurons),n_hidden_neurons=hidden_neurons,
        n_categories=1,epochs=100,batch_size=10,
        activation_function_type="LeakyRELU",
        errortype="MSE",solver="ADAM") #Create Neural Network
    testErr, trainErr, testR2, trainR2= Crossval_Neural_Network(5, nn, 1e-3, Lambda,X,y)
    assert trainErr<tol, "%.3f"%trainErr

test_NN_reg()
test_optimizers()
test_optimizers_ridge()
