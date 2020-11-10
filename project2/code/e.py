import numpy as np
from function_library import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



BENCHMARK = True
WRONG_CLASS_EX = False
ACC_V_LR = False
COMPARE_SCIKITLEARN = False

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



""" Proof of function, testing accuracy and examples of images that were wrongly classified """ 
if WRONG_CLASS_EX:
    #Performing Logistic regression and finding the weights and biases W and b
    Regressor = LogRegression(X_train, Y_train,n_epochs=10)
    W,b = Regressor.fit(eta = 0.001)

    #Getting the prediction and test accuracy of the model
    #by calling the predict method
    P = Regressor.predict(X_test,W,b)
    acc, Indx = Regressor.accuracy(P,Y_test, True)

    #Showing a random selection of wrong predictions
    random_indices = np.random.choice(Indx, size=4)

    for i, image in enumerate(X_test[random_indices]):
        y_test = np.argmax(Y_test[random_indices[i]])
        p = np.argmax(P[random_indices[i]])
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        reImage = image.reshape((8, 8))
        plt.imshow(reImage, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label = {}, Prediction = {}".format(y_test,p))
    plt.show()

""" Accuracy as a function of learning rate """
if ACC_V_LR:
    #Parameters start
    min_eta = -6
    max_eta = 2
    Nr_etas = 40
    Nr_epochs = 400
    #Parameters end

    #Scaling the data
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    #Initialization
    eta = np.logspace(min_eta,max_eta,Nr_etas)
    acc = np.zeros(Nr_etas)

    #Execution
    for i in range(Nr_etas):
        Reg = LogRegression(X_train, Y_train, n_epochs=Nr_epochs)
        W,b = Reg.fit(eta[i])
        acc[i] = Reg.accuracy(Reg.predict(X_test,W,b),Y_test)
        print("{} out of {}".format(i,Nr_etas))
    
    #Plotting
    plt.title("Test accuracy as a function of learning rate (#Epochs = {} )".format(Nr_epochs) )
    plt.plot(np.log10(eta),acc)
    plt.xlabel(r"Learning rate $\log_{10}(\eta)$")
    plt.ylabel(r"Test accuracy in $\%$ of correct predictions")
    plt.savefig("../figures/presentable_data/e_test_accuracy_over_learning_rate2.pdf")
    plt.show()

if COMPARE_SCIKITLEARN:
    NrRuns = 6

    Scipy_results = np.zeros(NrRuns)
    Own_results = np.zeros(NrRuns)

    # define inputs and labels
    X = digits.images
    y = digits.target

    #Structuring the data for analysis
    n_inputs = len(X)
    X = X.reshape(n_inputs, -1)

    for i in range(NrRuns):
        #Splitting into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        #Scaling the data
        scaler=StandardScaler()
        scaler.fit(X_train)
        X_train=scaler.transform(X_train)
        X_test=scaler.transform(X_test)

        #Fitting using ScikitLearn
        clf = LogisticRegression(penalty='none',
        tol=0.0001, 
        C=1.0, 
        fit_intercept=True, 
        solver='lbfgs', 
        max_iter=500, 
        multi_class='auto',
        n_jobs=None, 
        l1_ratio=None).fit(X_train,y_train)

        #Structuring the data for analysis
        y_train_HM = OneHotMatrix(y_train,10)
        y_test_HM = OneHotMatrix(y_test,10)

        #Performing Logistic regression and finding the weights and biases W and b
        Regressor = LogRegression(X_train, y_train_HM,n_epochs=500)
        W,b = Regressor.fit(eta = 0.0001)

        #Getting the prediction and test accuracy of the model
        #by calling the predict method
        P = Regressor.predict(X_test,W,b)
        Own_temp, Indx = Regressor.accuracy(P,y_test_HM, True)

        Sci_temp = clf.score(X_test,y_test)*100

        Scipy_results[i] = Sci_temp
        Own_results[i] = Own_temp

    print("Results:")
    print("Prediction accuracy- SciKitLearn = {} percent".format(Scipy_results))
    print("Prediction accuracy-         Own = {} percent".format(Own_results))


if BENCHMARK:
    NrRuns = 1

    Scipy_results = np.zeros(NrRuns)
    Own_results = np.zeros(NrRuns)

    # define inputs and labels
    X = digits.images
    y = digits.target

    #Structuring the data for analysis
    n_inputs = len(X)
    X = X.reshape(n_inputs, -1)

    """ PARAMETERS START """
    NrEpochs = 100
    LearningRate = 0.01

    """ PARAMETERS END """
    for i in range(NrRuns):
        #Splitting into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        #Scaling the data
        scaler=StandardScaler()
        scaler.fit(X_train)
        X_train=scaler.transform(X_train)
        X_test=scaler.transform(X_test)

        #Fitting using ScikitLearn
        clf = LogisticRegression(penalty='none',
        tol=0.0001, 
        C=1.0, 
        fit_intercept=True, 
        solver='lbfgs', 
        max_iter=NrEpochs, 
        multi_class='auto',
        n_jobs=None, 
        l1_ratio=None).fit(X_train,y_train)

        #Structuring the data for analysis
        y_train_HM = OneHotMatrix(y_train,10)
        y_test_HM = OneHotMatrix(y_test,10)

        #Performing Logistic regression and finding the weights and biases W and b
        Regressor = LogRegression(X_train, y_train_HM,n_epochs=NrEpochs)
        W,b = Regressor.fit(eta = LearningRate)

        #Getting the prediction and test accuracy of the model
        #by calling the predict method
        P = Regressor.predict(X_test,W,b)
        Own_temp, Indx = Regressor.accuracy(P,y_test_HM, True)

        Sci_temp = clf.score(X_test,y_test)*100

        Scipy_results[i] = Sci_temp
        Own_results[i] = Own_temp

    print("NrEpochs = {}, LearningRate = {}".format(NrEpochs,LearningRate))
    print("Results:")
    print("Prediction accuracy- SciKitLearn = {} percent".format(Scipy_results))
    print("Prediction accuracy-         Own = {} percent".format(Own_results))
        