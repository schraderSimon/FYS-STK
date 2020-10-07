import numpy as np
from function_library import *
from sklearn.linear_model import SGDRegressor
def test_optimizers():
    """
    This function tests wether all implemented Gradient Descent methods
    give the same result as the expected analytical result within a given tolerance
    (we choose 1e-2 as tolerance, given that these methods are not accurate)
    """
    x=np.random.randn(100)
    y=5*x+3
    X=DesignMatrix(x,2) # create a design matrix
    theta=np.array([3,5,0])
    sgd=SGD(X,y,500)
    theta_RMSprop=sgd.RMSprop(eta=0.001)
    theta_normal=sgd.simple_fit(eta=0.01)
    theta_decay=sgd.decay_fit(t1=50)
    tol=1e-2
    assert sum(abs(theta_RMSprop-theta)) <tol
    assert sum(abs(theta_normal-theta)) < tol
    assert sum(abs(theta_decay-theta)) < tol
test_optimizers()
