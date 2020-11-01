import numpy as np
from function_library import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Collect the  MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
X = digits.images
y = digits.target

#Structuring the data for analysis
n_inputs = len(X)
X = X.reshape(n_inputs, -1)
Y = OneHotMatrix(y,10)

#Splitting into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Performing Logistic regression and finding the weights and biases W and b
Regressor = LogRegression(X_train, Y_train,n_epochs=10)
W,b = Regressor.fit()

#Getting the test accuracy of the model, by calling the predict method
Regressor.accuracy(Regressor.predict(X_test,W,b),Y_test)

