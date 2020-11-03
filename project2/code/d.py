import numpy as np
from function_library import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


""" Data setup """ 
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

#Scaling the data
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

epochs=20000
n_hidden_neurons=[100,100]
n_hidden_layers=len(n_hidden_neurons)
n_categories=10
batch_size=200
Lambda=0
eta=0.01
activation_function_type_output="softmax"
errortype = "categorical"
solver="sgd"

nn=NeuralNetwork(X_train_scaled,Y_train,
    n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
    n_categories=n_categories,epochs=epochs,batch_size=batch_size,
    activation_function_type_output=activation_function_type_output,
    errortype=errortype,solver=solver) #Create Neural Network

nn.train()
prediction = nn.predict(X_test_scaled)
print("MLF: %f"%accuracy_score(OneHotToDigit(Y_test,10),prediction))
#print(prediction)
#print(OneHotToDigit(Y_test,10))
#testErr, trainErr, testR2, trainR2= Crossval_Neural_Network(5, nn, eta, Lambda,X,z)