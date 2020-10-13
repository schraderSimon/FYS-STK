import numpy as np
import sys
from numba import jit, int32, float32
from sklearn.preprocessing import StandardScaler
import sys
from numba.experimental import jitclass
def DesignMatrix(x,polydgree):
    """
    Input: Data x (array), the desired maximum degree (int)
    Output: The reular design matrix (exponent increases by one) for a polynomial of degree polydgree
    """
    X = np.zeros((len(x),polydgree+1))
    for i in range(polydgree+1):
        X[:,i] = x**i
    return X
@jit
def DesignMatrix_deg2(x,y,polydegree,include_intercept=False):
    """
    Input: x,y (created as meshgrids), the maximal polynomial degree, if the intercept should be included or not.

    The function creates a design matrix X for a 2D-polynomial.

    Returns: The design matrix X
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
    Input: A matrix X_matrix
    Output: The same matrix X with randomly shuffled rows
    much more computationally intensive than shuffling its indices..
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
    Input: A matrix X_matrix
    Output: An array containing the indeces of random shuffleing of the design matrix' row.
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
    Input: A design matrix X, the degree of k-fold Cross Validation k
    Output: A set of randomly chosen training and testing indeces
    which are, of course, distinct. Used in K-Fold Cross validation.
    2 2D arrays of length k containing the relevant data.

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
    For a given design matrix X and an outputs z,
    This function performs k-fold Cross Validation
    and calculates the RIDGE fit for a given Lambda for each k.
    The output is the estimate (average over k performs) for the mean square error.
    Input: X (matrix), z (vector), k (integer), lambda (double)
    Output: test error estimate (double)
    """
    #getting indices from Kfoldcross
    trainIndx, testIndx = KfoldCross(X,k)
    #init empty MSE array
    MSE_crossval = np.zeros(k)
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

    MSE_estimate = np.mean(MSE_crossval)
    #print("MSE Ridge")
    #print(MSE_estimate)
    return MSE_estimate

def R2(y_data,y_model):
    """
    Input: The original target data, the fitted data
    Output: The R2 value.
    """
    return 1- np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def MSE(y_data,y_model):
    """
    Input: Two arrays of equal length
    Output: The MSE between the two vectors
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def RidgeRegression(X_training,y_training,Lambda, include_beta_variance=True):

    """
    Input: The design matrix X, and the targets Y and a value for LAMBDA, and whether the variance should be included too.
    Output: The Ridge Regression beta and the variance (zero if include_beta_variance is zero)
    This was implemented as SVD.

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
#Implements Ridge Regression using Design matrix (X_training) training data of y (y_training)
#Returns the beta coeffs. and their variance

def LinearRegression(X_training,y_training,include_beta_variance=True):
    """Input: The design matrix X, and the targets Y
        Output: The OLS beta and the variance (zero if include_beta_variance is zero)
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
    def __init__(self,X,y,n_epochs=100,theta=0,batchsize = 1):
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
    def learning_schedule(self,t,t0,t1): #The learning schedule for the decay fit
        return t0/(t+t1)
    def calculateGradient(self,theta, index=-1):
        """Calculate the gradient at a random point X"""
        if index == -1:
            index=np.random.randint(self.n,size=self.batchsize) #If the index is not explicitely set, choose a random index

        xi= self.X[index]
        yi= self.y[index]

        gradients = 2/self.batchsize * xi.T @ ((xi @ theta).ravel()-yi)

        return gradients
    def simple_fit(self,eta=0.01):
        theta=self.theta
        for epoch in range(1,self.n_epochs+1): #For each epoch
            for i in range(self.MB): #For each minibatch
                theta=theta-eta*self.calculateGradient(theta);
        return theta.ravel()
    def decay_fit(self,t0=5,t1=50):
        theta=self.theta
        for epoch in range(1,self.n_epochs+1):
            for i in range(self.MB): #For each minibatch
                gradient=self.calculateGradient(theta);
                eta=self.learning_schedule(epoch*self.n+i,t0,t1)
                theta=theta-eta*gradient
        return theta.ravel()
    def RMSprop(self,eta=1e-2,cbeta=0.9, error=1e-8):
        theta=self.theta
        s=np.zeros_like(theta)
        for epoch in range(1,self.n_epochs+1):
            for i in range(self.MB): #For each minibatch
                gradient=self.calculateGradient(theta)
                s=cbeta*s+(1-cbeta)*(gradient*gradient)
                theta= theta-eta*gradient/np.sqrt(s+error)
        return theta.ravel()

class SGD_Ridge(SGD):
    def __init__(self,X,y,n_epochs=100,theta=0,batchsize = 1, Lambda=0.01):
        super().__init__(X,y,n_epochs,theta,batchsize);
        self.Lambda=Lambda;

    def calculateGradient(self,theta, index=-1):
        '''Calculate the gradient at a random point X for Ridge'''
        if index == -1:
            index=np.random.randint(self.n,size=self.batchsize) #If the index is not explicitely set, choose a random index

        xi= self.X[index]
        yi= self.y[index]

        gradients = (2/self.batchsize) * (xi.T @ ((xi @ theta).ravel()-yi)+self.Lambda*theta)

        return gradients
class NeuralNetwork():
    def __init__(
            self,
            X_data,
            Y_data,
            errortype="MSE",
            activation_function_type="sigmoid",
            activation_function_type_output="linear",
            gradienttype="RMSProp",
            n_categories=1,
            n_hidden_layers=1,
            n_hidden_neurons=[50]*1,
            epochs=2,
            batch_size=32,
            eta=1e-6,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.n_categories=n_categories
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_layers=n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.errortype=errortype
        self.activation_function_type=activation_function_type
        self.activation_function_type_output=activation_function_type_output
        self.gradienttype="RMSProp"
        self.create_biases_and_weights()
    def create_biases_and_weights(self):
        self.hidden_bias=[0]*self.n_hidden_layers
        self.hidden_weights=[1]*self.n_hidden_layers
        self.hidden_weights[0] = np.random.randn(self.n_features, self.n_hidden_neurons[0])
        self.hidden_bias[0] = np.zeros(self.n_hidden_neurons[0]) + 0.01
        for i in range(1,self.n_hidden_layers):
            self.hidden_weights[i]=np.random.randn(self.n_hidden_neurons[i-1], self.n_hidden_neurons[i])
            self.hidden_bias[i] = np.zeros(self.n_hidden_neurons[i]) + 0.01
        print(self.n_hidden_neurons[-1])
        self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01
    def activation_function(self,z,type=0):
        if type==0:
            type=self.activation_function_type
        if type==("linear"):
            return z
        if type==("sigmoid"):
            return 1/(1+np.exp(-z))
        if type==("RELO"):
            return np.max(z,0)
        if type==("LeakyRELO"):
            return max(z,0.01*z)
    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights[0]) + self.hidden_bias[0]
        self.a_h = self.activation_function(self.z_h,self.activation_function_type)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        self.probabilities = self.activation_function(self.z_o,self.activation_function_type_output)
    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activation_function(z_h,self.activation_function_type)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        return self.activation_function(z_o,self.activation_function_type_output)
    def elementwise_error(self):
        if self.errortype==("MSE"):
            return (self.probabilities-self.Y_data)**2
        if self.errortype==("categorical"):
            return self.probabilities - self.Y_data
    def backpropagation(self):
        error_output = (self.probabilities - self.Y_data)*1/self.batch_size
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
        print("Shapes")
        print(error_output.shape, self.output_weights.T.shape)
        print(self.a_h.shape)
        #sys.exit(1)
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)
        print(self.output_bias_gradient)
        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights[0]
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights[0]
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights[0] -= self.eta * self.hidden_weights_gradient
        self.hidden_bias[0] -= self.eta * self.hidden_bias_gradient
    def train(self):
        data_indices=np.arange(self.n_inputs)
        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints=np.random.choice(data_indices,size=self.batch_size,replace=False)
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                self.feed_forward()
                self.backpropagation()
    def predict(self,X):
        probabilities=self.predict_probabilities(X)
        return np.argmax(probabilities,axis=1)
    def predict_probabilities(self,X):
        probabilities=self.feed_forward_out(X)
        return probabilities

class NeuralNetwork2():
    def __init__(
            self,
            X_data,
            Y_data,
            errortype="MSE",
            activation_function_type="sigmoid",
            activation_function_type_output="linear",
            gradienttype="RMSProp",
            n_categories=1,
            n_hidden_layers=1,
            n_hidden_neurons=[50]*1,
            epochs=2,
            batch_size=32,
            eta=1e-6,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.n_categories=n_categories
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_layers=n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.errortype=errortype
        self.activation_function_type=activation_function_type
        self.activation_function_type_output=activation_function_type_output
        self.gradienttype="RMSProp"
        self.create_biases_and_weights()
    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons[0])
        self.hidden_bias = np.zeros(self.n_hidden_neurons[0]) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons[0], self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01
    def activation_function(self,z,type=0):
        if type==0:
            type=self.activation_function_type
        if type==("linear"):
            return z
        if type==("sigmoid"):
            return 1/(1+np.exp(-z))
        if type==("RELO"):
            return np.max(z,0)
        if type==("LeakyRELO"):
            return max(z,0.01*z)
    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activation_function(self.z_h,self.activation_function_type)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        self.probabilities = self.activation_function(self.z_o,self.activation_function_type_output)
    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activation_function(z_h,self.activation_function_type)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        return self.activation_function(z_o,self.activation_function_type_output)
    def elementwise_error(self):
        if self.errortype==("MSE"):
            return (self.probabilities-self.Y_data)**2
        if self.errortype==("categorical"):
            return self.probabilities - self.Y_data
    def backpropagation(self):
        error_output = (self.probabilities - self.Y_data)*1/self.batch_size
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
        print("Shapes")
        print(error_output.shape, self.output_weights.T.shape)
        print(self.a_h.shape)
        sys.exit(1)
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)
        print(self.output_bias_gradient)
        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient
    def train(self):
        data_indices=np.arange(self.n_inputs)
        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints=np.random.choice(data_indices,size=self.batch_size,replace=False)
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                self.feed_forward()
                self.backpropagation()
    def predict(self,X):
        probabilities=self.predict_probabilities(X)
        return np.argmax(probabilities,axis=1)
    def predict_probabilities(self,X):
        probabilities=self.feed_forward_out(X)
        return probabilities
