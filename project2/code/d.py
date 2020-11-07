import numpy as np
from function_library import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

""" What part of the code to run: Only one should be set to True lest there be unforseen consequences! Descriptions below """

BASIC = True
ACTIVATION_COMPARISON = True#False
ARCHITECTURE_COMPARISON = False
SCIKITLEARN = False#True

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
    eta=1e-3
    epochs=200
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
    accuracy_test = np.zeros(len(activation_function_type))
    accuracy_train = np.zeros(len(activation_function_type))

    #iterator is used for indexing the output array
    iterator = 0
    for activation in activation_function_type:# For every activation function in the list

        #Setup the network, the choice of X and Y doesn't matter as it's overwritten in
        # the Crossval_Neural_Network program
        nn=NeuralNetwork(X,Y,
            n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
            n_categories=n_categories,epochs=epochs,batch_size=batch_size,
            activation_function_type_output=activation_function_type_output,activation_function_type=activation,
            errortype=errortype,solver=solver) #Create Neural Network

        #Getting accuracy scores from Cross validation
        accuracy_test[iterator], accuracy_train[iterator] = Crossval_Neural_Network(k,nn,eta,Lambda,X,Y)

        iterator += 1
    #Printing outputs

    print("number of epochs = {}".format(epochs))
    print(activation_function_type)
    print("test accuracy")
    print(accuracy_test)
    print("train accuracy")
    print(accuracy_train)

if ARCHITECTURE_COMPARISON:
    #Initializing network
    epochs=100 #Number of epochs
    #Network architectures: Number of neurons in the hidden layers are given from left to right
    #every nested list specifies and additional architecture to be attempted in the loop below
    n_hidden_neurons_list=[[100,100,100,20],[50,50,20,20,20],[200],[100]]
    #Creates a list with the number of hidden layers for each architecture
    n_hidden_layers=[len(n_hidden_neurons_list[i]) for i in range(len(n_hidden_neurons_list))]
    n_categories=10
    batch_size=20
    Lambda=0.0001 #L2 regularization parameter
    eta=0.0001 #Learning rate
    activation_function_type_output="softmax"
    errortype = "categorical"
    solver="sgd"

    #For the hidden layers, tanh is the preferred activation function
    # as it is more numerically stable (for classification) than all others in our implementation
    activation_function_type= "tanh"

    #Number of folds used in kfold cross validation
    k = 4

    #initializing outputs
    accuracy_test = np.zeros(len(activation_function_type))
    accuracy_train = np.zeros(len(activation_function_type))

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
    print(" score: training,     score: testing")
