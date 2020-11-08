import numpy as np
import sys
from scipy.special import expit, softmax
from numba import jit, int32, float32
from sklearn.preprocessing import StandardScaler
import sys
from numba.experimental import jitclass
def DesignMatrix(x,polydgree):
    """
    Creates a design matrix of given degree based upon given input data.

    Parameters:
    x (array), the data
    polydegree (int), max. polynomial degree
    Returns:
    X (2D array), The regular design matrix (exponent increases by one) for a polynomial of degree polydgree
    """
    X = np.zeros((len(x),polydgree+1))
    for i in range(polydgree+1):
        X[:,i] = x**i
    return X
@jit
def DesignMatrix_deg2(x,y,polydegree,include_intercept=False):
    """
    The function creates a design matrix X for a 2D-polynomial.

    Parameters:
    x (array), meshgrid with x data,
    y (array), meshgrid with y data,
    polydegree (int), max. polynomial degree
    include_intercept (bool), if the intercept should be included or not.

    Returns:
    X (2D array), The design matrix X
    """
    adder=0 #The matrix dimension is increased by one if include_intercept is True
    p=round((polydegree+1)*(polydegree+2)/2)-1 #The total amount of coefficients
    if include_intercept:
        p+=1
        adder=1
    datapoints=len(x)
    X=np.zeros((datapoints,p))
    if include_intercept:
        X[:,0]=1
    X[:,0+adder]=x # Adds x on the first column
    X[:,1+adder]=y # Adds y on the second column
    """
    Create the design matrix:
    X[:,3] is x**2 * y**0, X[:,4] is x**1 * y **1,
    X[:,7] is x**2 * y ** 1 and so on. Up to degree degree
    """
    count=2+adder
    for i in range(2,polydegree+1):
        for j in range(i+1):
            X[:,count]=X[:,0+adder]**j*X[:,1+adder]**(i-j)
            count+=1;
    return X
def ShuffleRows(X_matrix):
    """
    Shuffle Rows in a matrix. Much more computationally intensive than shuffling its indices..

    Parameters:
    X_matrix (2D array)

    Returns:
    X_matrix (2D array), with randomly shuffled rows
    """
    length = len(X_matrix)
    for i in range(length):
        rand_index = np.random.randint(0,length-1)
        current_row = X_matrix[i,:]
        X_matrix[i,:] = X_matrix[rand_index,:]
        X_matrix[rand_index,:] = current_row
    return X_matrix
#Returns a list of randomly shuffled indices for a matrix/array
def ShuffleIndex(X_matrix):
    """
    Create n array containing the indeces of random shuffleing of the design matrix' row.

    Parameters:
    X_matrix (2D array)

    Returns:
    shuffled_indexs (array)

    """
    length = len(X_matrix)
    shuffled_indexs = list(range(0,length))
    for i in range(length):
        rand_val =np.random.randint(0,length-1)
        rand_index = shuffled_indexs[rand_val]
        current_index = shuffled_indexs[i]
        shuffled_indexs[i] = rand_index
        shuffled_indexs[rand_val] = current_index
    return shuffled_indexs
#Returns the training and testing indices for
#K-fold crossvalidation, given a design matrix & k-value
def KfoldCross(X_matrix,k):
    """
    Return training and testing indeces based upon a design matrix and a degree k for k-fold Cross validation.

    Parameters:
    X_matrix (2D array), the design matrx
    k (int),  the degree of k-fold Cross Validation

    Returns:
    training_indexes (2D array), distinct training indeces
    testing_indexes (2D array), distinct testing indeces

    """
    #Creating an array of shuffled indices
    shuffled_indexs = ShuffleIndex(X_matrix)
    #getting the length of the array
    list_length = len(shuffled_indexs)
    #getting the length of each partition up to a remainder
    partition_len = list_length//k
    #The number of remaining elements after the equal partitioning is noted
    remainder = list_length % k
    #creating empty arrays for the training and testing indices
    training_indexes = []
    testing_indexes = []
    #setting paramaters required for proper hanlding of remainders
    else_offset = 1
    current_index_end = 0
    current_index_start = 0
    for i in range(k):
        #when there's a remainder after partitioning,
        #increase partition length by 1 and
        #create partitions until remainder is 0
        if (remainder > 0):
            #setting start and stop indices for the partitons
            current_index_end = i*(partition_len+1)+partition_len+1
            current_index_start = i*(partition_len+1)
            testing_indexes.append(shuffled_indexs[current_index_start:current_index_end])
            training_indexes.append(shuffled_indexs[0:current_index_start] + shuffled_indexs[current_index_end:])
        #When the final remainder is included, changes the program
        #to only create partitions of the original length
        else:
            testing_indexes.append(shuffled_indexs[current_index_end:current_index_end + partition_len])
            training_indexes.append(shuffled_indexs[0:current_index_end] + shuffled_indexs[current_index_end+partition_len:])
            current_index_end += partition_len
            current_index_start += partition_len
        #subtracts 1 from the remainder each time
        remainder -= 1
    return training_indexes, testing_indexes
def KCrossValRidgeMSE(X,z,k,Lambda):
    """
    Perform K-Fold Cross validation for Ridge regression.


    For a given design matrix X and an outputs z,
    This function performs k-fold Cross Validation
    and calculates the RIDGE fit for a given Lambda for each k.
    The output is the estimate (average over k performs) for the mean square error.

    Parameters:
    X (matrix), the design matrix (non-scaled)
    z (array), the target data (non-scaled)
    k (int), the number of Cross Valdations (k-Fold)
    lambda (double), the regularization parameter

    Returns:
    MSE_estimate (double), test error
    MSE_train_estimate (double), train error
    """
    #getting indices from Kfoldcross
    trainIndx, testIndx = KfoldCross(X,k)
    #init empty MSE array
    MSE_crossval = np.zeros(k)
    MSE_crossval_train=np.zeros(k)
    #redef scaler, with_mean = True
    scaler = StandardScaler()
    for i in range(k):
        X_training = X[trainIndx[i],:]
        X_testing = X[testIndx[i],:]

        z_trainings = z[trainIndx[i]]
        z_testings = z[testIndx[i]]
        z_training=z_trainings-np.mean(z_trainings)
        z_testing=z_testings-np.mean(z_trainings)
        #Scale X
        scaler.fit(X_training)
        X_training_scaled = scaler.transform(X_training)
        X_testing_scaled = scaler.transform(X_testing)
        #perform Ridge regression
        beta, beta_variance = RidgeRegression(X_training_scaled,z_training,Lambda)
        #print(beta)
        z_training_fit = X_training_scaled @ beta
        z_testing_fit = X_testing_scaled @ beta
        #calculate MSE for each fold
        MSE_crossval[i] = MSE(z_testing,z_testing_fit)
        MSE_crossval_train[i] = MSE(z_training,z_training_fit)
    MSE_estimate = np.mean(MSE_crossval)
    MSE_train_estimate=np.mean(MSE_crossval_train)

    return MSE_estimate, MSE_train_estimate

def accuracy_score(y_data, y_model):
    """The accuracy score"""
    return np.sum(y_data==y_model)/len(y_data)

def R2(y_data,y_model):
    """The R2 value of a fit"""
    return 1- np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def MSE(y_data,y_model):
    """The Mean Square Error (MSE) of a fit."""
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def RidgeRegression(X_training,y_training,Lambda, include_beta_variance=True):

    """
    Performs Ridge Regression. Implemented as SVD

    Parameters:
    X_training (2D array), The design matrix X,
    y_training (array), the targets Y
    Lambda (double), regularization parameter,
    include_beta_variance (bool), whether the variance should be included too.

    Returns:
    beta (array), the Ridge regression beta
    beta_variance (2D array),The R.R. var (0 if include_beta_variance is False)

    """
    I = np.eye(len(X_training[0,:]))
    if include_beta_variance:
        inverse_matrix = np.linalg.inv(X_training.T @ X_training+Lambda*I)
        beta_variance = np.diagonal(inverse_matrix)
    else:
        beta_variance=0
    u, s, vh = np.linalg.svd(X_training, full_matrices=False)
    smat=np.zeros((vh.shape[0],u.shape[1]))
    for i in range(len(s)):
        smat[i][i]=s[i]
    beta= vh.T @ (np.linalg.inv(smat.T@smat+(I*Lambda)) @ smat.T) @ u.T @ y_training
    return beta, beta_variance

def LinearRegression(X_training,y_training,include_beta_variance=True):
    """
    Performs OLS. Implemented as SVD

    Parameters:
    X_training (2D array), The design matrix X,
    y_training (array), the targets Y
    include_beta_variance (bool), whether the variance should be included too.

    Returns:
    beta (array), the OLS beta
    beta_variance (2D array),The OLS var (0 if include_beta_variance is False)

    """
    if include_beta_variance:
        inverse_matrix = np.linalg.inv(X_training.T @ X_training)
        beta_variance = np.diag(inverse_matrix)
    else:
        beta_variance=0
    u, s, vh = np.linalg.svd(X_training, full_matrices=False)
    beta= vh.T @ np.linalg.inv(np.diag(s)) @ u.T @ y_training
    return beta, beta_variance

class SGD:
    """
    Stochastic Gradient Descent without Regularization

    Parameters:
    X (2D array), the design matrix
    y (1D array), the target data
    n_epochs (int), the number of epochs
    theta (array), the initial guess for theta
    batchsize (int), the batch size
    n (int), number of rows in the design matrix
    p (int), number of columns in the design matrix
    MB (int), number of minibatches
    Methods:
    reset, resets theta to some random number
    learning_schedule(t(double),t0(double), t1(double)),
        calculates the learning schadule
    calculateGradient(theta (array),index (1D array)),
        calculates the gradient based on a given index
    simple_fit(eta (double)), calculate theta using simple SGD
    decay_fit(t0 (double), t1(double)), calculate eta using decay SGD
    RMSprop(eta (double),cbeta (double), error (double)),
        calculate theta using RMSProp
    ADAM(eta (double),beta_1 (double), beta_2( double), error (double)),
        calculate theta using ADAM
    """
    def __init__(self,X,y,n_epochs=100,theta=0,batchsize = 1):
        """
        Parameters:
        X (2D array), the design matrix
        y (1D array), the target data
        n_epochs (int), the number of epochs
        theta (array), the initial guess for theta
        batchsize (int), the batch size
        """
        self.n_epochs=n_epochs;
        self.X=X;
        self.y=y;
        self.n=len(X) #Number of rows
        self.p=len(X[0]) #number of columns
        self.batchsize=batchsize
        self.MB=int(self.n/self.batchsize) #number of minibatches

        if theta==0:
            self.theta=np.random.randn(self.p,1).ravel() #Create start guess for theta if theta is not given as parameter
        else:
            self.theta=theta.ravel()
    def reset(self):
        self.theta=np.random.randn(self.p,1).ravel()
    def learning_schedule(self,t,t0,t1): #The learning schedule for the decay fit
        return t0/(t+t1)
    def calculateGradient(self,theta, index=-1):
        """Calculate the gradient at a random point X"""
        if index == -1:
            index=np.random.randint(self.n,size=self.batchsize) #If the index is not explicitely set, choose a random index

        xi= self.X[index]
        yi= self.y[index]

        gradients = 2/self.batchsize * xi.T @ ((xi @ theta).ravel()-yi) #The formula to calculate the Gradient for MSE

        return gradients
    def simple_fit(self,eta=0.01):
        theta=self.theta
        for epoch in range(1,self.n_epochs+1): #For each epoch
            for i in range(self.MB): #For each minibatch
                theta=theta-eta*self.calculateGradient(theta);
        return theta.ravel()
    def decay_fit(self,t0=5,t1=50):
        """Returns theta using the decay fit from the slides"""
        theta=self.theta
        for epoch in range(1,self.n_epochs+1):
            for i in range(self.MB): #For each minibatch
                gradient=self.calculateGradient(theta);
                eta=self.learning_schedule(epoch*self.n+i,t0,t1)
                theta=theta-eta*gradient
        return theta.ravel()
    def RMSprop(self,eta=1e-2,cbeta=0.9, error=1e-8):
        """Returns theta using RMSProp"""
        theta=self.theta
        s=np.zeros_like(theta)
        for epoch in range(1,self.n_epochs+1):
            for i in range(self.MB): #For each minibatch
                gradient=self.calculateGradient(theta)
                s=cbeta*s+(1-cbeta)*(gradient*gradient)
                theta= theta-eta*gradient/np.sqrt(s+error)
        return theta.ravel()

    def ADAM(self,eta=1e-3,beta_1=0.9,beta_2=0.99, error=1e-8):
        """Returns theta using the ADAM optimizer"""
        theta=self.theta
        s=np.zeros_like(theta)
        m=np.zeros_like(theta)
        s_hat=np.zeros_like(theta)
        m_hat=np.zeros_like(theta)
        for epoch in range(1,self.n_epochs+1):
            for i in range(1,self.MB+1): #For each minibatch
                gradient=self.calculateGradient(theta)
                m=beta_1*m+(1-beta_1)*gradient
                s=beta_2*s+(1-beta_2)*(gradient*gradient)
                m_hat = m/(1-beta_1**i)
                s_hat = s/(1-beta_2**i)
                theta= theta-eta*m_hat/(np.sqrt(s_hat))
                if (np.any(np.isnan(theta))):
                    sys.exit(1)
        return theta.ravel()


class SGD_Ridge(SGD):
    """
    Stochastic Gradient Descent with Regularization. Inherits from SGD class.

    Additional Parameters:
        Lambda (double), regularization parameter

    Updated methods:
        calculateGradient(theta (array),index (1D array)),
            calculates the gradient based on a given index
    """
    def __init__(self,X,y,n_epochs=100,theta=0,batchsize = 1, Lambda=0.01):
        super().__init__(X,y,n_epochs,theta,batchsize); #SGD initializer
        self.Lambda=Lambda; #set lambda

    def calculateGradient(self,theta, index=-1):
        '''Calculate the gradient at a random point X for Ridge'''
        if index == -1:
            index=np.random.randint(self.n,size=self.batchsize) #If the index is not explicitely set, choose a random index

        xi= self.X[index]
        yi= self.y[index]
        #The formula to calculate the Gradient for the Ridge Cost function
        gradients = (2/self.batchsize) * (xi.T @ ((xi @ theta).ravel()-yi)+self.Lambda*theta)


        return gradients

def OneHotMatrix(y,NrClasses):
    """
    Input: Vector y containing integers between 0 and y_max
    Integer: NrClasses indicating the number of classes
    Output: One- Hot matrix of shape (#datapoints, #classes)
    """
    n = len(y)
    M = np.zeros((n,NrClasses))
    for i in range(n):
        M[i,y[i]] = 1
    return M

def OneHotToDigit(M,NrClasses):
    """
    Input: Matrix M containing integers between 0 and y_max
    Integer: NrClasses indicating the number of classes
    Output: array of digits corresponding to One- Hot matrix
    """
    n = len(M)
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.argmax(M[i,:])
    return y

class LogRegression:
    """
    Logistic regression for classification using SoftMax and the CrossEntropy loss function

    Takes as input: X- Design matrix with predictors on the columns and datapoints on the rows
                    Y- OneHot- matrix labeling the data in X
                    n_epochs- the number of epochs
                    batchsize
                    Lambda- L2 regularization parameter

    Methods: fit: performs a fit of the data using a fixed learning rate
                  and SGD with the Cross Entropy loss function
                  Returns:
                  Weight- matrix and bias- vector trained on the inputs X and Y

            predict: Predicts the output as a one- hot matrix using test data X_test
                    returns:
                    One- Hot Matrix (#datapoints, #classes) P with predictions

            accuracy: Gives the accuracy of the model on test data as a percentage of correct guesses
                    returns:
                    accuracy as a percentage, indices of failed predictions
    """
    def __init__(self,X,Y,n_epochs=100,batchsize = 1, Lambda = 0.0):
        self.n_epochs=n_epochs
        self.X=X
        self.Y=Y
        self.Lambda = Lambda
        self.NrCategories = len(Y[0]) # number of categories
        self.n=len(X) #Number of datapoints in the design matrix
        self.NrPredictors=len(X[0]) #number of predictors in the design matrix
        self.batchsize=batchsize
        self.MB=int(self.n/self.batchsize) #number of minibatches

        #Initializing random weights and biases
        self.b = np.random.rand(self.NrCategories,1)
        self.W = np.random.rand(self.NrCategories, self.NrPredictors)

    def SoftMax(self,z):
        #More numerically stable SoftMax function
        z_reduced = z-max(z)
        #Checks if z_reduced returns NaN and exits if it does
        if np.isnan(z_reduced[0]):
            sys.exit[0]
        return np.exp(z_reduced)/(np.sum(np.exp(z_reduced))+10**(-12))

    def CrossEntropyLoss(self,y,a):
        #Cross entropy cost function
        return -1*np.sum(np.dot(y,np.log(a)))

    def FeedForward(self,x,W,b):
        #calculates the activation function from weights and biases using softmax
        z = np.matmul(W,x) + b
        a = self.SoftMax(z)
        return a

    def calculateGradient(self,W,b,x,y,a):
        #calculates the gradient of the Cross Entropy cost function
        NrCategories = self.NrCategories

        M = np.matmul(a, np.ones((1,NrCategories))) * (np.identity(NrCategories) - np.matmul(np.ones((NrCategories, 1)), a.T))

        da = - y/(a+10**(-15))
        dz = np.matmul(M, da)
        dW = 2/self.batchsize *(dz * x.T) + 2/self.batchsize *self.Lambda*W
        db = 2/self.batchsize *(dz.copy())+ 2/self.batchsize *self.Lambda*b
        return dW, db

    def fit(self,eta=0.01):
        W = self.W
        b = self.b
        X = self.X
        Y = self.Y

        for epoch in range(1,self.n_epochs+1): #For each epoch
            for i in range(self.MB): #For each minibatch
                index=np.random.randint(self.n,size=self.batchsize) #selecting random index from the minibatch
               #Reshaping data
                x= X[index].reshape(self.NrPredictors,1)
                y= Y[index].reshape(self.NrCategories,1)

                #feed forward
                a = self.FeedForward(x,W,b)

                # getting gradients
                dW, db = self.calculateGradient(W,b,x,y,a)

                # Calculate new weights and biases
                W = W - eta*dW
                b = b - eta*db
        return W, b

    def predict(self,X_test,W,b):
        #Makes a prediction using test- design matrix, weights and biases
        n = len(X_test)
        #initialization of output
        P = np.zeros((n,self.NrCategories))
        for i in range(n):
            z = np.matmul(W,X_test[i].reshape(self.NrPredictors,1))+b
            #getting the output through the SoftMax activation function
            a = self.SoftMax(z)
            #inserting a 1 in the zero- matrix to get One- Hot output
            P[i,np.argmax(a)] = 1
        return P

    def accuracy(self,P,Y,ReturnIndex = False):
        #gets the accuracy of the model, using the predictions and test- labels
        n = len(Y)
        #initialization of outputs
        Wrong = 0
        Indx = []

        for i in range(n):
            #If prediction is wrong error will equal 1
            #if it's right, wrong will equal 0
            error = np.max(P[i,:]-Y[i,:])
            #Collecting every error in variable 'Wrong'
            #'Wrong' will finally be the number of wrong predictions
            Wrong += error
            #If the model has made a wrong prediction, append the index
            if error > 0:
                Indx.append(i)
        #get the accuracy as a percentage
        acc = (1-int(Wrong)/n)*100
        print("Accuracy is {:.2f} percent".format(acc))
        #If ReturnIndex is true, this function also returns the indices
        if ReturnIndex:
            return acc, Indx
        else:
            return acc

class NeuralNetwork():
    """A simple FFNN for Regression and Classification, using MSE for Regression
    and CrossEntropy for Classification.

    Parameters:
    X_data_full (2D array), the design matrix
    Y_data_full (1D or 2D array), the target data (2D for onehot)
    errortype (string), The tye of error to be reduced (MSE or categorical)
    activation_function_type (string), the activation function for hidden layers
    activation_function_type_output (string), the act. func. for outp. layer
    solver (string), the solver to be used (sgd, RMSProp or ADAM)
    n_categories (int), the number of categories (1 for Regression)
    n_hidden_layers (int), the number of hidden layers
    n_hidden_neurons (list), the number of neurons per hidden layer
    epochs (int), the number of epochs
    batch_size (int), the batch size
    eta (double), the learning rate eta
    lmbd (double), the regularization parameter lambda
    linear_coeff (double), the slope for the linear output function (1 default)
    n_inputs (int), The number of inputs
    n_features (int), the number of features
    iterations (int), the number of iterations per epoch
    hidden_bias (list of arrays), bias for the hidden layers
    hidden_weights (list of 2D arrays), weights for the hidden layers
    output_bias (array), bias for the output layer
    output_weights (2D array), weights for the output layer

    z_h (list of 2D arrays), the z for the hidden layers
    z_o (2D array), the z for the output layer
    a_h (list), the activation for the hidden layers
    probabilities (array), the output data
    output_weights_gradient (2D array), the gradient for the output weights
    output_bias_gradient (array), the gradient for the output bias
    hidden_weights_gradient (list of 2D arrays), the gradient of the hidden weights
    hidden_bias_gradient (list of 1D arrays), the gradient for the hidden bias

    X_data (2D array), the datapoints to base a train step in
    Y_data (1D array), the datapoints to base a train step in

    iterator (int), the number of iterations (for ADAM)
    s (list of 1D & 2D arrays), the s for ADAM & RMSProp
    cbeta (double), the beta in RMSProp
    beta_1 (double), the beta1 in ADAM
    beta_2 (double), the beta2 in ADAM
    Methods:
    change_matrix(X (2D array), y (1D array)):
        updates X_data_full and y_data full,
        resets weights, biases, relevant sizes, RMSProp & ADAM
    update_parameters_reset(eta (float), lambda(float)):
        updates eta and lambda
        resets weights, biases, relevant sizes, RMSProp & ADAM
    create_biases_and_weights():
        set up initial weights and biases
    setUpRMSProp():
        sets up arrays for RMSProp
    setUpADAM():
        sets up arrays for ADAM
    activation_function(z (array), type(string or 0):
        returns the activation of an input z,
        depending on the chosen type of activation function
    derivative (a (array), z(array), type(string or 0)):
        returns the derivative of a given activation function,
        depending on the chosen type of activation function
        may depend on a or z
    feed_forward():
        Feeds forward the NN
    feed_forward_out():
        Feeds forward the (pretrained) neural network, creates output based on input
        Parameters:
            X (2D array), The design matrix
        Returns:
            y (1D/2D array), the prediction
    elementwise_error():
        Returns the error in the output (derivative of error function)
    error_function(y_data (array), y_model (array)):
        Parameters:
        y_data (1D arrray), the target
        y_model (1D arrray), the fitted data
        Returns:
        The quality of the fit (MSE and R2 for Regression, acc. score for class.)
    backpropagation():
        Calculates the gradients for the biases and weights and updates them (calls solve)
    solve():
        Updates biases and weights
    train():
        Runs one feed forward and one back propagation
    predict_probabilities(X (2D array)):
        Calculates the output based on the input design matrix (Regression)
    predict(X (2D array)):
        Calculates the most likely element based on prediction
    backpropagation():
        Calculates
    """
    def __init__(
            self,
            X_data,
            Y_data,
            errortype="MSE", #The type of error to be reduced
            activation_function_type="sigmoid", #The activation function for the hidden layer
            activation_function_type_output="linear", #The activation function for the output layer
            solver="sgd", #The solver to be used (sgd, RMSProp or ADAM)
            n_categories=1, #The number of categories (1 for Regression, more for classification)
            n_hidden_layers=1, #The number of hidden layers
            n_hidden_neurons=[50]*1, #A list containing the number of neurons per layer (0 to n-1)
            epochs=2, #The number of epochs
            batch_size=32, #The Batch size
            eta=1e-6, #The learning rate
            lmbd=0.0, #The regularization parameter lambda
            linear_coeff=1): #The stigningstall for hte linear activation function. Should be 1, I suppose?

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.n_categories=n_categories
        self.n_inputs = X_data.shape[0] #The number of inputs
        self.n_features = X_data.shape[1] # The number of n_features
        self.n_hidden_layers=n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size #The number of iterations per epoch
        self.eta = eta
        self.lmbd = lmbd
        self.errortype=errortype
        self.activation_function_type=activation_function_type
        self.activation_function_type_output=activation_function_type_output
        self.solver=solver
        self.linear_coeff=linear_coeff
        self.create_biases_and_weights() #Creates biases and weights based on some paper
        if self.solver=="RMSProp":
            self.setUpRMSProp() #Intialises s for RMSProp
        if self.solver=="ADAM":
            self.setUpADAM() #Intialises s,m,beta_1 and beta_2 for ADAM

    def change_matrix(self,X,y):
        """Changes the dataset to X, y and upgrades the relevant parameters"""
        self.X_data_full=X
        self.Y_data_full=y
        self.n_inputs=X.shape[0]
        self.n_features=X.shape[1]
        self.iterations = self.n_inputs // self.batch_size #The number of iterations per epoch
        self.create_biases_and_weights() #Creates biases and weights based on some paper
        if self.solver=="RMSProp":
            self.setUpRMSProp() #Intialises s for RMSProp
        if self.solver=="ADAM":
            self.setUpADAM() #Intialises s,m,beta_1 and beta_2 for ADAM

    def update_parameters_reset(self,eta,lmbd):
        """Update eta and lambda and reset weights and biases"""
        self.eta=eta
        self.lmbd=lmbd
        self.create_biases_and_weights() #In order to avoid "previous" approximation, everything is reset
        if self.solver=="RMSProp":
            self.setUpRMSProp() #Reset s
        if self.solver=="ADAM":
            self.setUpADAM() #Resets s,m,beta_1 and beta_2 for ADAM

    def create_biases_and_weights(self):
        """Sets biases and weights to random (weights) or small (bias) numbers"""
        self.hidden_bias=[0]*self.n_hidden_layers #an empty list of length n_hidden_layers
        self.hidden_weights=[1]*self.n_hidden_layers #an empty list of length n_hidden_layers
        #Set up the weights for the first hidden layer with gaussian distributed numbers with sigma=2/self.n_inputs self.batchsize
        self.hidden_weights[0] = np.random.randn(self.n_features, self.n_hidden_neurons[0])*np.sqrt(2/self.n_inputs)
        #Set up the biases for the first hidden layer as 0.001
        self.hidden_bias[0] = np.zeros(self.n_hidden_neurons[0]) + 0.001
        for i in range(1,self.n_hidden_layers):
            #Set up the biases and weights for all hidden layers the same way as for the first layer
            self.hidden_weights[i]=np.random.randn(self.n_hidden_neurons[i-1], self.n_hidden_neurons[i])*np.sqrt(2/self.n_inputs)
            self.hidden_bias[i] = np.zeros(self.n_hidden_neurons[i]) + 0.001
        #Set up the biases and weights for all the output layer the same way as for the first layer
        self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_categories)*np.sqrt(2/self.n_inputs)
        self.output_bias = np.zeros(self.n_categories) + 0.001

    def setUpRMSProp(self):
        """set the array s for RMSProp so that each layer has it's own s-array"""
        self.s=[]
        self.s.append(np.zeros_like(self.output_weights))
        self.s.append(np.zeros_like(self.output_bias))
        for i in range(self.n_hidden_layers):
            self.s.append(np.zeros_like(self.hidden_weights[i]))
            self.s.append(np.zeros_like(self.hidden_bias[i]))
        self.cbeta=0.9

    def setUpADAM(self):
        """sets up the arrays for ADAM"""
        self.iterator = 0
        self.s=[]; self.s.append(np.zeros_like(self.output_weights)); self.s.append(np.zeros_like(self.output_bias))
        self.m=[]; self.m.append(np.zeros_like(self.output_weights)); self.m.append(np.zeros_like(self.output_bias))
        for i in range(self.n_hidden_layers):
            self.m.append(np.zeros_like(self.hidden_weights[i]))
            self.m.append(np.zeros_like(self.hidden_bias[i]))
            self.s.append(np.zeros_like(self.hidden_weights[i]))
            self.s.append(np.zeros_like(self.hidden_bias[i]))
        self.beta_1=0.9
        self.beta_2=0.99

    def activation_function(self,z,type=0):
        """The different activation functions"""
        if type==0:
            type=self.activation_function_type
        if type==("linear"):
            return self.linear_coeff*z
        if type==("sigmoid"):
            sigmoid = expit(z)
            return sigmoid

        if type==("tanh"):
            return 2*self.activation_function(2*z,type="sigmoid") -1
        if type==("RELU"):
            return np.maximum(z,0,z)
        if type==("LeakyRELU"):
            return np.maximum(z,0.01*z,z)
            #if z is larger than zero, we get z, if it's below zero, then 0.01z > z.
        if type=="softmax":
            """
            #Implementation of a more numerically stable softmax function
            z_reduced = z-z.max()
            return np.exp(z_reduced)/(np.sum(np.exp(z_reduced))+10**(-12))
            """
            m=np.max(z)
            exp_term=np.exp(z-m) #This is to avoid problems of too large numbers
            if np.isnan(z[0,0]):
                print("Something is going wrong here (expect wrong results)")
            returnval= exp_term / np.sum(exp_term, axis=1, keepdims=True)
            return returnval
            #return returnval
    def derivative(self,a,z,type=0):
        """The derivative of the activation function"""
        #takes both a and z as arguments because some things are more efficient with a than z
        if type==0:
            type=self.activation_function_type #If no activation type is given, use the object's
        if type==("linear"):
            return self.linear_coeff
        if type==("sigmoid"):
            return a*(1-a)
        if type=="tanh":
            return 1-a*a
        if type=="RELU":
            deriv=np.copy(z)
            deriv[deriv<0]=0
            deriv[deriv>0]=1
            return deriv
        if type=="LeakyRELU":
            deriv=np.copy(z)
            deriv[deriv>0]=1
            deriv[deriv<0]=0.01
            return deriv
        if type == "softmax":
            ####################### NOT TESTED ########################
            f = self.activation_function(z,type="softmax")
            return np.matmul(f,np.identity(len(f))-f)

    def feed_forward(self):
        # feed-forward for training
        self.z_h=[0]*self.n_hidden_layers
        self.a_h=[0]*self.n_hidden_layers
        self.z_h[0] = np.matmul(self.X_data, self.hidden_weights[0]) + self.hidden_bias[0] #Calculate z for 1st. hidden layer
        self.a_h[0] = self.activation_function(self.z_h[0],self.activation_function_type) #Calculate activation for 1st. hidden layer
        for i in range(1,self.n_hidden_layers):
            #calculate z and activation for 2nd. to last hidden layer
            self.z_h[i]=np.matmul(self.a_h[i-1], self.hidden_weights[i]) + self.hidden_bias[i]
            self.a_h[i] = self.activation_function(self.z_h[i],self.activation_function_type)

        self.z_o = np.matmul(self.a_h[-1], self.output_weights) + self.output_bias #Calculate z for output layer

        self.probabilities = self.activation_function(self.z_o,self.activation_function_type_output) #Calculate output

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h=[0]*self.n_hidden_layers
        a_h=[0]*self.n_hidden_layers
        z_h[0] = np.matmul(X, self.hidden_weights[0]) + self.hidden_bias[0] #Calculate z for 1st. hidden layer
        a_h[0] = self.activation_function(z_h[0],self.activation_function_type) #Calculate activation for 1st. hidden layer
        for i in range(1,self.n_hidden_layers):
            #calculate z and activation for 2nd. to last hidden layer
            z_h[i]=np.matmul(a_h[i-1], self.hidden_weights[i]) + self.hidden_bias[i]
            a_h[i] = self.activation_function(z_h[i],self.activation_function_type)
        z_o = np.matmul(a_h[-1], self.output_weights) + self.output_bias
        return self.activation_function(z_o,self.activation_function_type_output) #Return output

    def elementwise_error(self):
        """The type of gradient for the error measure"""
        if self.errortype==("MSE"):
            return (self.probabilities-self.Y_data)*1/self.batch_size #The type of error
        if self.errortype==("categorical"): #CrossEntropy
            return (self.probabilities - self.Y_data)#/self.batch_size

    def error_function(self,y_data,y_model):
        """The error function for MSE & categorial  """
        if self.errortype==("MSE"):
            return MSE(y_data,y_model), R2(y_data,y_model)
        if self.errortype==("categorical"):
            return accuracy_score(y_data,y_model)

    def solve(self):
        """Update weights and biases based on the updated gradients"""
        if self.solver=="sgd": #SGD
            if self.lmbd > 0.0:
                self.output_weights_gradient += self.lmbd * self.output_weights #Add regularization
            #Update output layer
            self.output_weights -= self.eta * self.output_weights_gradient
            self.output_bias -= self.eta * self.output_bias_gradient
            for i in range(self.n_hidden_layers):
                #Update hidden layers
                if self.lmbd > 0.0:
                    self.hidden_weights_gradient[i] += self.lmbd * self.hidden_weights[i]
                self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient[i]
                self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient[i]

        elif self.solver=="RMSProp": #RMSProp
            cbeta=self.cbeta
            self.s[0]=cbeta*self.s[0]+(1-cbeta)*(self.output_weights_gradient*self.output_weights_gradient) #Update s
            self.s[1]=cbeta*self.s[1]+(1-cbeta)*(self.output_bias_gradient*self.output_bias_gradient) #Update s
            #Update output layer
            self.output_weights -= self.eta * self.output_weights_gradient/np.sqrt(self.s[0]+1e-8)
            self.output_bias -= self.eta * self.output_bias_gradient/np.sqrt(self.s[1]+1e-8)
            for i in range(self.n_hidden_layers):
                #Update hidden layers
                if self.lmbd > 0.0:
                    self.hidden_weights_gradient[i] += self.lmbd * self.hidden_weights[i]
                self.s[2+i*2]=cbeta*self.s[2+i*2]+(1-cbeta)*(self.hidden_weights_gradient[i]*self.hidden_weights_gradient[i])
                self.s[3+i*2]=cbeta*self.s[3+i*2]+(1-cbeta)*(self.hidden_bias_gradient[i]*self.hidden_bias_gradient[i])
                self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient[i]/np.sqrt(self.s[2+i*2]+1e-8)
                self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient[i]/np.sqrt(self.s[3+i*2]+1e-8)

        elif self.solver=="ADAM": #ADAM Optimizer
            beta_1=self.beta_1
            beta_2=self.beta_2
            self.m[0]=beta_1*self.m[0]+(1-beta_1)*self.output_weights_gradient #Update m
            self.m[1]=beta_1*self.m[1]+(1-beta_1)*self.output_bias_gradient#Update m
            self.s[0]=beta_2*self.s[0]+(1-beta_2)*(self.output_weights_gradient*self.output_weights_gradient) #Update s
            self.s[1]=beta_2*self.s[1]+(1-beta_2)*(self.output_bias_gradient*self.output_bias_gradient) #Update s
            #Update output layer
            self.output_weights -= self.eta * (self.m[0]/(1-beta_1**(self.iterator+1)))/(np.sqrt(self.s[0]/(1-beta_2**(self.iterator+1)))+1e-8)
            self.output_bias -= self.eta * (self.m[1]/(1-beta_1**(self.iterator+1)))/(np.sqrt(self.s[1]/(1-beta_2**(self.iterator+1)))+1e-8)
            for i in range(self.n_hidden_layers):
                #Update hidden layers
                if self.lmbd > 0.0:
                    self.hidden_weights_gradient[i] += self.lmbd * self.hidden_weights[i]
                self.m[2+i*2]= beta_1*self.m[2+i*2]+(1- beta_1)* self.hidden_weights_gradient[i]
                self.m[3+i*2]= beta_1*self.m[3+i*2]+(1- beta_1)* self.hidden_bias_gradient[i]
                self.s[2+i*2]= beta_2*self.s[2+i*2]+(1- beta_2)*(self.hidden_weights_gradient[i]*self.hidden_weights_gradient[i])
                self.s[3+i*2]= beta_2*self.s[3+i*2]+(1- beta_2)*(self.hidden_bias_gradient[i]*self.hidden_bias_gradient[i])
                self.hidden_weights[i] -= self.eta * (self.m[2+i*2]/(1-beta_1**(self.iterator+1)))/(np.sqrt(self.s[2+i*2]/(1-beta_2**(self.iterator+1)))+1e-8)
                self.hidden_bias[i] -= self.eta * (self.m[3+i*2]/(1-beta_1**(self.iterator+1)))/(np.sqrt(self.s[3+i*2]/(1-beta_2**(self.iterator+1)))+1e-8)
            self.iterator += 1

    def backpropagation(self):
        error_output = self.elementwise_error() #The error to calculating the gradient
        self.output_weights_gradient = np.matmul(self.a_h[-1].T, error_output) #calculate the gradient for the ouptput
        self.output_bias_gradient = np.sum(error_output, axis=0) #calculate the gradient for the ouptput bias


        error_hidden=[0]*self.n_hidden_layers #empty array

        #Calculate the error for the last hidden layer
        error_hidden[-1] = np.matmul(error_output, self.output_weights.T) *self.derivative(self.a_h[-1],self.z_h[-1],self.activation_function_type)
        for i in range(self.n_hidden_layers-1,0,-1):
            #Calculate error for the other hidden layers based on the previous hidden layer
            error_hidden[i-1]= np.matmul(error_hidden[i], self.hidden_weights[i].T)*self.derivative(self.a_h[i-1],self.z_h[i-1],self.activation_function_type)

        #Calculate the gradients for the hidden layers
        self.hidden_weights_gradient=[0]*self.n_hidden_layers#creating lists of length n_hidden_layers
        self.hidden_bias_gradient=[0]*self.n_hidden_layers
        self.hidden_weights_gradient[0] = np.matmul(self.X_data.T, error_hidden[0])
        self.hidden_bias_gradient[0] = np.sum(error_hidden[0], axis=0)
        for i in range(1,self.n_hidden_layers,1):
            self.hidden_bias_gradient[i] = np.sum(error_hidden[i], axis=0)
            self.hidden_weights_gradient[i] = np.matmul(self.a_h[i-1].T, error_hidden[i])

        #Update weights and biases
        self.solve()
    def train(self):
        data_indices=np.arange(self.n_inputs) #Indexes for the number of inputs
        for i in range(self.epochs): #Fore each epoch
            for j in range(self.iterations): #For the number of iterations per epoch
                chosen_datapoints=np.random.choice(data_indices,size=self.batch_size,replace=False) #choose random datapoints
                self.X_data = self.X_data_full[chosen_datapoints] #Update datapoints
                self.Y_data = self.Y_data_full[chosen_datapoints] #Update datapoints
                self.feed_forward()
                self.backpropagation()
    def predict(self,X):
        #Categorization: Return probabilities
        probabilities=self.predict_probabilities(X)
        return np.argmax(probabilities,axis=1)
    def predict_probabilities(self,X):
        #Prediction: Return probabilities
        probabilities=self.feed_forward_out(X)
        return probabilities

def Crossval_Neural_Network(k, nn, eta, Lambda,X,z):
        """input: The number of cross validations k,
                    the neural network nn,
                the learning rate eta,
                the regularization parameter lambda,
                the (unscaled) data X, z
            output: test and train error as well as R2 score for training and testing"""
        """Here, the X is the full set"""
        trainIndx, testIndx = KfoldCross(X,k) #Get random indices
        Error_test = np.zeros(k); R2_test=np.zeros(k)
        Error_train=np.zeros(k); R2_train=np.zeros(k)
        scaler=StandardScaler()
        if nn.errortype=="categorical":
            scaler = StandardScaler(with_mean=True,with_std=False)
        if nn.errortype=="MSE":
            scaler = StandardScaler()
        for i in range(k): #For the munber of cross validations
            """Seperate in training and testing sets, scale"""
            X_training = X[trainIndx[i],:]
            X_testing = X[testIndx[i],:]
            z_trainings = z[trainIndx[i]]
            z_testings = z[testIndx[i]]
            if nn.errortype=="MSE":
                z_training=z_trainings-np.mean(z_trainings)
                z_testing=z_testings-np.mean(z_trainings)
            else:
                z_training=z_trainings
                z_testing=z_testings
            #Scale X
            scaler.fit(X_training)
            X_training_scaled = scaler.transform(X_training)
            X_testing_scaled = scaler.transform(X_testing)

            """For regression problems"""
            if nn.errortype=="MSE":
                z_training=z_training.reshape((X_training_scaled.shape[0],1))
                z_testing=z_testing.reshape((X_testing_scaled.shape[0],1))
            nn.change_matrix(X_training_scaled,z_training) #Update dataset
            nn.update_parameters_reset(eta=eta,lmbd=Lambda) #Update parameters
            nn.train() #Train the set

            if i==1:
                #print(prediction_test)
                #break;
                pass

            if nn.errortype=="MSE":
                """Calculate train and test error"""
                prediction_train=nn.predict_probabilities(X_training_scaled)
                prediction_test=nn.predict_probabilities(X_testing_scaled)

                Error_train[i],R2_train[i] = nn.error_function(z_training,prediction_train)
                Error_test[i],R2_test[i]=nn.error_function(z_testing,prediction_test)

            """For classification problems"""
            if nn.errortype=="categorical":
                #Calculate train and test error
                prediction_test=nn.predict(X_testing_scaled)
                prediction_train=nn.predict(X_training_scaled)

                Error_test[i] = accuracy_score(OneHotToDigit(z_testings,nn.n_categories),prediction_test)
                Error_train[i] = accuracy_score(OneHotToDigit(z_trainings,nn.n_categories),prediction_train)
                #It's called Error_test, but its just the accuracy

        if nn.errortype=="MSE":
            error_train_estimate = np.mean(Error_train);R2_train_estimate=np.mean(R2_train)
            error_test_estimate = np.mean(Error_test);R2_test_estimate=np.mean(R2_test)
            return error_test_estimate, error_train_estimate, R2_test_estimate, R2_train_estimate

        if nn.errortype=="categorical":
            error_test_estimate = np.mean(Error_test)
            error_train_estimate = np.mean(Error_train)
            return error_test_estimate, error_train_estimate

from sklearn.model_selection import KFold as SKFold
from sklearn.neural_network import MLPRegressor
def CrossVal_Regression(k,eta,Lambda,X,z,activation_function_type,solver,n_hidden_neurons,epochs):
    """Cross Validation using Scikit Learn's MLPRegressor

    Parameters:
    Everything that is needed to create an MLPObject

    Returns:
        error estimates and R2 estimates for train and test error
    """

    kf=SKFold(n_splits=k,shuffle=True)
    Error_test = np.zeros(k); R2_test=np.zeros(k)
    Error_train=np.zeros(k); R2_train=np.zeros(k)
    scaler = StandardScaler()
    trainIndx, testIndx = KfoldCross(X,k) #Get random indices
    for i in range(k): #For the munber of cross validations
        """Seperate in training and testing sets, scale"""
        X_training = X[trainIndx[i],:]
        X_testing = X[testIndx[i],:]
        z_trainings = z[trainIndx[i]]
        z_testings = z[testIndx[i]]
        z_training=z_trainings-np.mean(z_trainings)
        z_testing=z_testings-np.mean(z_trainings)
        #Scale X
        scaler.fit(X_training)
        X_training_scaled = scaler.transform(X_training)
        X_testing_scaled = scaler.transform(X_testing)
        z_training=z_training.reshape((X_training_scaled.shape[0],1))
        z_testing=z_testing.reshape((X_testing_scaled.shape[0],1))
        regr=MLPRegressor(learning_rate_init=eta,max_iter=epochs,solver=solver,alpha=Lambda,
            hidden_layer_sizes=n_hidden_neurons,activation=activation_function_type).fit(X_training_scaled,z_training.ravel())

        prediction_train=regr.predict(X_training_scaled)
        prediction_test=regr.predict(X_testing_scaled)

        Error_train[i],R2_train[i] =MSE(z_training.ravel(),prediction_train), R2(z_training.ravel(),prediction_train)
        Error_test[i],R2_test[i]=MSE(z_testing.ravel(),prediction_test), R2(z_testing.ravel(),prediction_test)
    error_train_estimate = np.mean(Error_train);R2_train_estimate=np.mean(R2_train)
    error_test_estimate = np.mean(Error_test);R2_test_estimate=np.mean(R2_test)
    return error_test_estimate, error_train_estimate, R2_test_estimate, R2_train_estimate

from sklearn.neural_network import MLPClassifier

""" Kfold Cross validation of a Scikit-Learn implementation of a Feed Forward Neural network Classifier.
For the purpose of accurately gauging its accuracy """
def CrossVal_SKLClassifier(X,Y,k,
        hidden_layer_sizes=(200,100,50,20 ),
        activation='tanh',
        alpha=0.0001,
        batch_size=20,
        learning_rate='constant',
        learning_rate_init=0.01,
        max_iter=5,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        momentum=0.9,
        nesterovs_momentum=False,
        early_stopping=False,
        n_iter_no_change=10):

    """
    Takes as input the complete dataset X and labels Y (as one- hot matrix), and the number of folds k
    The final input arguments are parameters for initializing the neural network. This excempts parameters
    related to the solver, which is set to 'sgd' regardless.
    """

    #initializing outputs
    Accuracy_test = np.zeros(k)
    Accuracy_train = np.zeros(k)
    #set up scaler
    scaler = StandardScaler()

    #retrieve training and testing indices
    trainIndx, testIndx = KfoldCross(X,k)

    for i in range(k): #For the munber of cross validations

        #setup data for this fold with the indices gotted above
        X_training = X[trainIndx[i],:]
        X_testing = X[testIndx[i],:]
        Y_training = Y[trainIndx[i]]
        Y_testing = Y[testIndx[i]]

        #Scaling X ( We don't scale labels in classification )
        scaler.fit(X_training)
        X_training_scaled = scaler.transform(X_training)
        X_testing_scaled = scaler.transform(X_testing)

        # initializing NN with Stochastic gradient descent as solver
        mlp = MLPClassifier(hidden_layer_sizes=(200,100,50,20 ),
        activation=activation,
        solver='sgd',
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change)

        #Fitting to training data
        mlp.fit(X_training_scaled,Y_training)

        #Retrieving accuracy scores
        Accuracy_train[i] = mlp.score(X_training_scaled,Y_training)
        Accuracy_test[i] = mlp.score(X_testing_scaled,Y_testing)

    #Taking the mean
    Accuracy_train_estimate = np.mean(Accuracy_train)
    Accuracy_test_estimate = np.mean(Accuracy_test)
    return Accuracy_test_estimate, Accuracy_train_estimate
