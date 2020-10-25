import numpy as np
import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from function_library import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
X = digits.images
y = digits.target

n_inputs = len(X)
X = X.reshape(n_inputs, -1)

n_estimators = len(X[0])
n_classes = 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


theta = np.random.rand(n_estimators, 10)

sgd = SGD(X_train,y_train)
parameters = sgd.RMSprop()

y_model = np.dot(X_test,parameters)

def IndicatorFunction(y,k):
    out = 0
    if y==k:
        out = 1
    return out

def SoftMax(theta, x,k):
    return np.exp(np.dot(theta[:,k],x))/(np.sum([np.dot(theta[:,i],x ) for i in range(n_classes) ]))

def CostFunction(theta,y,X):
    cost = 0
    for i in range(n_inputs):
        for k in range(n_classes):
            cost += IndicatorFunction(y[i],k)*np.log(SoftMax(theta,X[i],k))
    return -cost

def Gradient_SoftMax(X,y,theta,k):
    out = np.zeros(len(X[0]))
    for i in range(n_inputs):
      out  += X[i]*(IndicatorFunction(y[i],k)-SoftMax(theta,X[i],k))
    return out

n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

m = 100
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ ((xi @ theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients
print("theta from own sdg")
print(theta)