import numpy as np
import random as rn
def bootstrap_bias(test,predict,n):
    bias=np.zeros(n)
    for i in range(n):
        bias[i]=np.mean((test-np.mean(predict[:,i]))**2)
    bias=np.zeros(len(test))
    for i in range(len(test)):
        bias[i]=np.mean((np.mean(predict[i,:])-test[i])**2)
    return np.mean(bias)

def bootstrap_variance(test,predict,n):
    variance=np.zeros(n)
    for i in range(n):
        variance[i]=np.mean((predict[:,i]-np.mean(predict[:,i]))**2)
    variance=np.zeros(len(test))
    for i in range(len(test)):
        variance[i]=np.mean((predict[i,:]-np.mean(predict[i,:]))**2)
    print("variance:", variance[i])
    return np.mean(variance)

def bootstrap_MSE(test,predict,n):
    error=np.zeros(n)
    for i in range(n):
        error[i]=np.mean((predict[:,i]-test)**2)
    print("error: ",error[i])
    return np.mean(error)

def resample(X,y):
    """
    Input: Arrays X (actually a matrix), y of equal size
    Output: A bootstrap resample of X and y
    """
    amount_datapoints=len(y)
    resample=np.random.randint(amount_datapoints,size=amount_datapoints)
    #print(resample)
    X_resampled=X[:][resample]
    y_resampled=y[resample]
    return X_resampled, y_resampled

def R2(y_data,y_model):
    return 1- np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

#returns a regular (exponent increases by one) design matrix from data x and polynomial of degree n
def DesignMatrix(x,polydgree):
    X = np.zeros((len(x),polydgree+1))
    for i in range(polydgree+1):
        X[:,i] = x**i
    return X

#Implements Ridge Regression using Design matrix (X_training) training data of y (y_training),
#  the lambda coefficient, and the degree of the approximating polynomial.
#Returns the beta coeffs. and their variance
def RidgeRegression(X_training,y_training,Lambda,PolyApproxDegree):
    I = np.eye(PolyApproxDegree)
    inverse_matrix = np.linalg.inv(X_training.T @ X_training+Lambda*I)
    beta_variance = np.diagonal(inverse_matrix)
    beta = inverse_matrix @ X_training.T @ y_training
    return beta, beta_variance

#Implements Ridge Regression using Design matrix (X_training) training data of y (y_training)
#Returns the beta coeffs. and their variance
def LinearRegression(X_training,y_training):
    inverse_matrix = np.linalg.inv(X_training.T @ X_training)
    beta_variance = np.diagonal(inverse_matrix)
    beta = inverse_matrix @ X_training.T @ y_training
    return beta, beta_variance

#Returns the evaluation of a polynomial with the coefficients beta at point x
def Coeff_to_Poly(x,beta):
    poly = 0
    for i in range(len(beta)):
        poly += beta[i]*x**i
    return poly

#The Franke Function.
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Design Matrix for a 2D-polynomial.

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

#import numpy as np
#import random as rn

#Shuffles the rows of a matrix....
# much more computationally intensive than shuffling its indices..
def ShuffleRows(X_matrix):
    length = len(X_matrix)
    for i in range(length):
        rand_index = rn.randint(0,length-1)
        current_row = X_matrix[i,:]
        X_matrix[i,:] = X_matrix[rand_index,:]
        X_matrix[rand_index,:] = current_row
    return X_matrix
#Returns a list of randomly shuffled indices for a matrix/array
def ShuffleIndex(X_matrix):
    length = len(X_matrix)
    shuffled_indexs = list(range(0,length))
    for i in range(length):
        rand_val =rn.randint(0,length-1)
        rand_index = shuffled_indexs[rand_val]
        current_index = shuffled_indexs[i]
        shuffled_indexs[i] = rand_index
        shuffled_indexs[rand_val] = current_index
    return shuffled_indexs

#First attempt at K fold crossvalidation... for posterity
"""
def KfoldCross(X_matrix,k):
    shuffled_indexs = ShuffleIndex(X_matrix)
    list_length = len(shuffled_indexs)
    partition_len = list_length//k
    remainder = list_length % k
    partitions = []
    print(list_length)###############delete
    print(remainder)#################delete
    for i in range(k):
        partitions.append(shuffled_indexs[i*partition_len:(i+1)*partition_len])
        if (remainder > 1):
            partitions[i].extend(shuffled_indexs[-remainder:-remainder+1])
            remainder -= 1
    partitions[-1].append(shuffled_indexs[-1])

    testing_indexes = np.zeros(k)
    training_indexes = np.zeros(k)
    for i in range(k):
        testing_indexes[i].extend(partitions[i])
        for j in range(k):
            if (j != i):
                training_indexes[i].extend(partitions[j])
    return training_indexes, testing_indexes

"""


#Returns the training and testing indices for 
#K-fold crossvalidation, given a design matrix & k-value
def KfoldCross(X_matrix,k):
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
