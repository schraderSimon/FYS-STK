import numpy as np
from function_library import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns
sns.set()

""" What part of the code to run: Only one should be set to True lest there be unforseen consequences! Descriptions below """

BASIC = False#True
ACTIVATION_COMPARISON = False
ARCHITECTURE_COMPARISON = False
SCIKITLEARN = True

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
    scaler=StandardScaler(with_mean=True,with_std=False)
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    #neural network parameters
    eta=1e-3
    epochs=500
    n_hidden_neurons=[100,100,50,50]
    n_hidden_layers=len(n_hidden_neurons)
    n_categories=10
    batch_size=100
    Lambda=0.0001
    activation_function_type_output="softmax"
    activation_function_type="LeakyRELU"
    errortype = "categorical"
    solver="ADAM"
    #Setting up the network
    nn=NeuralNetwork(X_train_scaled,Y_train,
        n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
        n_categories=n_categories,epochs=epochs,batch_size=batch_size,eta=eta,lmbd=Lambda,
        activation_function_type=activation_function_type,
        activation_function_type_output=activation_function_type_output,
        errortype="categorical")

    #Training the network
    nn.train()
    #Making a prediction on the test data
    prediction = nn.predict(X_test_scaled)

    #Prints the accuracy on the test data ( No cross- validation )
    print("MLF: %f"%accuracy_score(OneHotToDigit(Y_test,10),prediction))

""" Hidden Layer activation function comparison """
if ACTIVATION_COMPARISON:

    #neural network parameters
    etas=[0.00001,0.0001,0.001,0.01,0.1]
    epochs=[5,50,100,200]
    n_hidden_neurons=[100,100,50,50]
    n_hidden_layers=len(n_hidden_neurons)
    n_categories=10
    batch_size=100
    Lambda=0.0001

    activation_function_type_output="softmax"
    errortype = "categorical"
    solver="ADAM"

    # Setting up a list of Different Hidden layer activation functions to be tested
    activation_function_type= ["tanh","sigmoid","LeakyRELU","RELU"]

    #Number of folds used in kfold- cross validation
    k = 4

    #Initializing outputs
    accuracy_test = np.zeros((len(activation_function_type),len(epochs),len(etas)))
    accuracy_train = np.zeros((len(activation_function_type),len(epochs),len(etas)))

    #iterator is used for indexing the output array
    iterator = 0
    for activation in activation_function_type:# For every activation function in the list
        for i in range(len(epochs)):
            for j in range(len(etas)):

                #Setup the network, the choice of X and Y doesn't matter as it's overwritten in
                # the Crossval_Neural_Network program
                nn=NeuralNetwork(X,Y,
                    n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
                    n_categories=n_categories,epochs=epochs[i],batch_size=batch_size,
                    activation_function_type_output=activation_function_type_output,activation_function_type=activation,
                    errortype=errortype,solver=solver) #Create Neural Network

                #Getting accuracy scores from Cross validation
                accuracy_test[iterator,i,j], accuracy_train[iterator,i,j] = Crossval_Neural_Network(k,nn,etas[j],Lambda,X,Y)

        iterator += 1
    
    plt.figure(figsize=(10,10))
    ax1=plt.subplot(421)
    ax2=plt.subplot(422)
    ax3=plt.subplot(423)
    ax4=plt.subplot(424)
    ax1.set_title(r"tanh")
    ax2.set_title(r"sigmoid")
    ax3.set_title(r"LeakyRELU")
    ax4.set_title(r"RELU")
    ax1.ticklabel_format(useOffset=False)
    ax2.ticklabel_format(useOffset=False)
    ax3.ticklabel_format(useOffset=False)
    ax4.ticklabel_format(useOffset=False)

    sns.heatmap(accuracy_test[0,:,:], xticklabels=etas, yticklabels=epochs, annot=True, ax=ax1, cmap="viridis")
    sns.heatmap(accuracy_test[1,:,:], xticklabels=etas, yticklabels=epochs, annot=True, ax=ax2, cmap="viridis")
    sns.heatmap(accuracy_test[2,:,:], xticklabels=etas, yticklabels=epochs, annot=True, ax=ax3, cmap="viridis")
    sns.heatmap(accuracy_test[3,:,:], xticklabels=etas, yticklabels=epochs, annot=True, ax=ax4, cmap="viridis")

    ax1.set_ylabel(r"#epochs");ax2.set_ylabel(r"#epochs");ax3.set_ylabel(r"#epochs");ax4.set_ylabel(r"#epochs")
    ax1.set_xlabel(r"$\eta$"); ax2.set_xlabel(r"$\eta$"); ax3.set_xlabel(r"$\eta$");ax4.set_xlabel(r"$\eta$")
    plt.tight_layout()
    plt.savefig("../figures/Activation_function_d.pdf")
    plt.show()
    
    
    #Printing outputs

    print("number of epochs = {}".format(epochs))
    print(activation_function_type)
    print("test accuracy")
    print(accuracy_test)
    print("train accuracy")
    print(accuracy_train)

if ARCHITECTURE_COMPARISON:
    #Initializing network
    epochs=150 #Number of epochs
    #Network architectures: Number of neurons in the hidden layers are given from left to right
    #every nested list specifies and additional architecture to be attempted in the loop below
    n_hidden_neurons_list=[[100,100,100,20],[50,50,20,20,20],[200],[100]]
    #Creates a list with the number of hidden layers for each architecture
    n_hidden_layers=[len(n_hidden_neurons_list[i]) for i in range(len(n_hidden_neurons_list))]
    n_categories=10
    batch_size=100
    Lambda=0.0001 #L2 regularization parameter
    eta=0.001 #Learning rate
    activation_function_type_output="softmax"
    errortype = "categorical"
    solver="ADAM"

    #For the hidden layers we use LeakyRELU as the activation function as this performed best 
    # given a network with 4 hidden layers
    activation_function_type= "LeakyRELU"

    #Number of folds used in kfold cross validation
    k = 4

    #initializing outputs
    accuracy_test = np.zeros(len(n_hidden_neurons_list))
    accuracy_train = np.zeros(len(n_hidden_neurons_list))

    #For every network architecture
    for i in range(len(n_hidden_neurons_list)):
        #print(n_hidden_neurons_list[i])

        #Setup network
        nn=NeuralNetwork(X,Y,
            n_hidden_layers=n_hidden_layers[i],n_hidden_neurons=n_hidden_neurons_list[i],
            n_categories=n_categories,epochs=epochs,batch_size=batch_size,
            activation_function_type_output=activation_function_type_output,activation_function_type=activation_function_type,
            errortype=errortype,solver=solver) #Create Neural Network

        #Getting accuracies from cross validation
        accuracy_test[i], accuracy_train[i] = Crossval_Neural_Network(k,nn,eta,Lambda,X,Y)

    print("number of epochs = {}".format(epochs))
    #print(activation_function_type)
    print("test accuracy")
    print(accuracy_test)
    print("train accuracy")
    print(accuracy_train)

if SCIKITLEARN:

    #Compare methods for the same k- value
    k = 4 #Making sure that Scikitlearn and our NN has the same size training set

    # Calls function that implements Scikit Learn's FFNN Classifier: MLPClassifier in Kfold Cross Validation
    #It returns the model accuracy on the testing and training data

    print(CrossVal_SKLClassifier(X,Y,k))
    print(" score: testing,     score: training")
