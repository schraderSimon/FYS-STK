import numpy as np
from function_library import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASIC = False
ACTIVATION_COMPARISON = True

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




""" Basic demonstration of Classifying neural network """
if BASIC:

    #Splitting into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    #Scaling the data
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    epochs=800
    n_hidden_neurons=[20]
    n_hidden_layers=len(n_hidden_neurons)
    n_categories=10
    batch_size=20
    Lambda=0
    eta=0.1
    activation_function_type_output="softmax"
    activation_function_type="LeakyRELU"
    errortype = "categorical"
    solver="ADAM"

    nn=NeuralNetwork(X_train_scaled,Y_train,
        n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
        n_categories=n_categories,epochs=epochs,batch_size=batch_size,
        activation_function_type_output=activation_function_type_output,activation_function_type=activation_function_type,
        errortype=errortype,solver=solver) #Create Neural Network

    nn.train()
    prediction = nn.predict(X_test_scaled)
    print("MLF: %f"%accuracy_score(OneHotToDigit(Y_test,10),prediction))

""" Hidden Layer activation function comparison """
if ACTIVATION_COMPARISON:

    epochs=10
    n_hidden_neurons=[20,50]
    n_hidden_layers=len(n_hidden_neurons)
    n_categories=10
    batch_size=20
    Lambda=0.0001
    eta=0.05
    activation_function_type_output="softmax"
    errortype = "categorical"
    solver="sgd"

    activation_function_type= ["LeakyRELU","RELU","tanh","sigmoid"]

    k = 4

    accuracy = np.zeros(len(activation_function_type))

iterator = 0
for activation in activation_function_type:
    
    nn=NeuralNetwork(X,Y,
        n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
        n_categories=n_categories,epochs=epochs,batch_size=batch_size,
        activation_function_type_output=activation_function_type_output,activation_function_type=activation,
        errortype=errortype,solver=solver) #Create Neural Network

    accuracy[iterator] = Crossval_Neural_Network(k,nn,eta,Lambda,X,Y)
    iterator += 1
print("number of epochs = {}".format(epochs))
print(activation_function_type)
print(accuracy)



"""
    for activation in activation_function_type:
        for i in range(NrAverageRuns):
            acc_temp = 0
            #Splitting into train and test data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

            #Scaling the data
            scaler=StandardScaler()
            scaler.fit(X_train)
            X_train_scaled=scaler.transform(X_train)
            X_test_scaled=scaler.transform(X_test)

            nn=NeuralNetwork(X_train_scaled,Y_train,
                n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
                n_categories=n_categories,epochs=epochs,batch_size=batch_size,
                activation_function_type_output=activation_function_type_output,activation_function_type=activation,
                errortype=errortype,solver=solver) #Create Neural Network

            nn.train()
            prediction = nn.predict(X_test_scaled)
            acc_temp += accuracy_score(OneHotToDigit(Y_test,10),prediction)
        accuracy[iterator] = acc_temp/NrAverageRuns
"""


