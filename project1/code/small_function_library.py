import numpy as np
import random as rn
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from numba import jit
lasso_tol=0.01
lasso_iterations=10*1e5
def bootstrap_r2(test,predict,n):
    """Input: The target data test (1D array),
    The prediction data predict (2D array of length n),
    the length n, which is the amount of bootstraps.
    Output: The averaged R2-value.
    """
    r2=np.zeros(n)
    for i in range(n):
        r2[i]=R2(test,predict[:,i])
    return np.mean(r2)
def bootstrap_bias(test,predict,n):
    """
    Input: The target data test (1D array),
    The prediction data predict (2D array of length n),
    the length n, which is the amount of bootstraps.
    Output: The averaged bias.
    """
    bias=np.zeros(n)
    for i in range(n):
        bias[i]=np.mean((test-np.mean(predict[:,i]))**2)
    bias=np.zeros(len(test))
    for i in range(len(test)):
        bias[i]=np.mean((np.mean(predict[i,:])-test[i])**2)
    return np.mean(bias)

def bootstrap_variance(test,predict,n):
    """
    Input: The target data test (1D array),
    The prediction data predict (2D array of length n),
    the length n, which is the amount of bootstraps.
    Output: The averaged variance.
    """
    variance=np.zeros(n)
    for i in range(n):
        variance[i]=np.mean((predict[:,i]-np.mean(predict[:,i]))**2)
    variance=np.zeros(len(test))
    for i in range(len(test)):
        variance[i]=np.mean((predict[i,:]-np.mean(predict[i,:]))**2)
    return np.mean(variance)

def bootstrap_MSE(test,predict,n):
    """
    Input: The target data test (1D array),
    The prediction data predict (2D array of length n),
    the length n, which is the amount of bootstraps.
    Output: The averaged MSE.
    """
    error=np.zeros(n)
    for i in range(n):
        error[i]=np.mean((predict[:,i]-test)**2)
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

def DesignMatrix(x,polydgree):
    """
    Input: Data x (array), the desired maximum degree (int)
    Output: The reular design matrix (exponent increases by one) for a polynomial of degree polydgree
    """
    X = np.zeros((len(x),polydgree+1))
    for i in range(polydgree+1):
        X[:,i] = x**i
    return X

#Implements Ridge Regression using Design matrix (X_training) training data of y (y_training),
#  the lambda coefficient, and the degree of the approximating polynomial.
#Returns the beta coeffs. and their variance
#IMPORTANT: This form of Ridge regression assumes that X_training has been centered
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

def LASSORegression(X_training,y_training,Lambda,tol=lasso_tol,iter=lasso_iterations):
    """Input: The design matrix X, and the targets Y and a value for LAMBDA
        Output: The LASSO Regression beta. Uses scikit-learn.
    """
    clf = linear_model.Lasso(fit_intercept=False,alpha=Lambda,tol=tol,max_iter = iter)
    clf.fit(X_training,y_training)
    return clf.coef_ #beta.

#Returns the evaluation of a polynomial with the coefficients beta at point x
def Coeff_to_Poly(x,beta):
    poly = 0
    for i in range(len(beta)):
        poly += beta[i]*x**i
    return poly

#The Franke Function.
def FrankeFunction(x,y):
    """
    Input: x and y
    Output: The Franke Function evaluated at x and y.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Design Matrix for a 2D-polynomial.
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
        rand_index = rn.randint(0,length-1)
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
        rand_val =rn.randint(0,length-1)
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

def KCrossValMSE(X,z,k,scaling = True):
    """
    This function performs k-fold Cross Validation
    and calculates the OLS fit for a given Lambda for each k.
    The output is the estimate (average over k performs) for the mean square error.
    Input: X (matrix), z (vector), k (integer)
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
        #If Scaling == True
        if (scaling):
            #Scale X
            scaler.fit(X_training)
            X_training_scaled = scaler.transform(X_training)
            X_testing_scaled = scaler.transform(X_testing)
            X_training_scaled[:,0] = 1
            X_testing_scaled[:,0] = 1
            #perform OLS regression
            beta, beta_variance = LinearRegression(X_training_scaled,z_training)
            z_training_fit = X_training_scaled @ beta
            z_testing_fit = X_testing_scaled @ beta
            #calculate MSE for each fold
            MSE_crossval[i] = MSE(z_testing,z_testing_fit)
            continue
        #If scaling == False
        beta, beta_variance = LinearRegression(X_training,z_training)
        z_training_fit = X_training @ beta
        z_testing_fit = X_testing @ beta

        MSE_crossval[i] = MSE(z_testing,z_testing_fit)
    MSE_estimate = np.mean(MSE_crossval)
    return MSE_estimate

def KCrossValLASSOMSE(X,z,k,Lambda,tol=lasso_tol,iter=lasso_iterations):
    """
    For a given design matrix X and an outputs z,
    This function performs k-fold Cross Validation
    and calculates the LASSO fit for a given Lambda.
    The output is the estimate for the mean square error.
    Input: X (matrix), z (vector), k (integer), lambda (double)
    Output: test error estimate (double)
    """
    #getting indices from Kfoldcross
    trainIndx, testIndx = KfoldCross(X,k)
    #init empty MSE array
    MSE_crossval = np.zeros(k)
    MSE_crossval_OLS = np.zeros(k)
    #redef scaler, with_mean = True
    for i in range(k):
        X_training = X[trainIndx[i],:]
        X_testing = X[testIndx[i],:]
        z_trainings = z[trainIndx[i]]
        z_testings = z[testIndx[i]]
        z_training=z_trainings-np.mean(z_trainings)
        z_testing=z_testings-np.mean(z_trainings)
        scaler = StandardScaler()
        scaler.fit(X_training)
        z_training=z_training#-np.mean(z_training)
        z_testing=z_testing#-np.mean(z_training)
        X_training_scaled =scaler.transform(X_training)
        X_testing_scaled = scaler.transform(X_testing)
        clf = clf = linear_model.Lasso(fit_intercept=False,alpha=Lambda,tol=tol,max_iter = iter)
        clf.fit(X_training_scaled,z_training)
        z_training_fit=clf.predict(X_training_scaled)
        z_testing_fit=clf.predict(X_testing_scaled)
        OLS=linear_model.LinearRegression(fit_intercept=True)
        OLS.fit(X_training_scaled,z_training)
        z_testing_fit_OLS=OLS.predict(X_testing_scaled)
        MSE_crossval[i] = MSE(z_testing,z_testing_fit)
        MSE_crossval_OLS[i]=MSE(z_testing,z_testing_fit_OLS)
    MSE_estimate = np.mean(MSE_crossval)
    MSE_estimate_OLS=np.mean(MSE_crossval_OLS)
    #print("LAMBDA: %f LASSO: %f OLS:%f"%(Lambda,MSE_estimate,MSE_estimate_OLS))
    return MSE_estimate

def KCrossValOLSMSE(X,z,k):
    """
    For a given design matrix X and an outputs z,
    This function performs k-fold Cross Validation
    and calculates the OLS fit.
    The output is the estimate (average over k performs) for the mean square error.
    Input: X (matrix), z (vector), k (integer)
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
        #perform regression
        beta, beta_variance = LinearRegression(X_training_scaled,z_training)
        #print(beta)
        z_training_fit = X_training_scaled @ beta
        z_testing_fit = X_testing_scaled @ beta
        #calculate MSE for each fold
        MSE_crossval[i] = MSE(z_testing,z_testing_fit)

    MSE_estimate = np.mean(MSE_crossval)
    #print("MSE Ridge")
    #print(MSE_estimate)
    return MSE_estimate


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

#(Not used)-- averages an array over a given interval to smooth out jagged plots
def ArraySmoother(Arr, interval):
    NrIntervals = len(Arr)//interval
    k=0
    smoothed = np.zeros(len(Arr))
    for i in range(NrIntervals):
        smoothed[i*interval:(i+1)*interval] = np.mean(Arr[i*interval:(i+1)*interval])
        k+=1
    smoothed[k*interval:] = np.mean(Arr[k*interval:])
    return smoothed
@jit
def fit_func(beta,x,y,polydegree,mean,inverse_var,include_intercept=False):
    """
    THIS DOES NOT WORK
    Input: Beta (array), x(array), y(array), polynomial degree (int),
    scaling mean (float), inverse std (float), wether the output should be included
    Output: z (float) fitted using the parameters
    """
    adder=0 #The matrix dimension is increased by one if include_intercept is True
    p=round((polydegree+1)*(polydegree+2)/2)-1 #The total amount of coefficients
    if include_intercept:
        p+=1
        adder=1
    func=np.zeros(p)
    if include_intercept:
        func[0]=1
    func[0+adder]=(x-mean[0+adder])*inverse_var[0+adder] # Adds x on the first column
    func[1+adder]=(y-mean[1+adder])*inverse_var[1+adder] # Adds y on the second column
    count=2+adder
    xpot=[x**j for j in range(polydegree+1)]
    ypot=[y**j for j in range(polydegree+1)]
    for i in range(2,polydegree+1):
        for j in range(i+1):
            func[count]=(xpot[j]*ypot[i-j]-mean[count])*inverse_var[count]
            count+=1;
    z=func @ beta
    return z
def fit_terrain(x,y,beta,scaler,mean_valz,degree=5,scaling=1):
    """
    THIS DOES NOT WORK
    Input: x(array), y(array), the choosen scaler, the mean value of the array z,polynomial degree (int), the scaling factor for the image
    Output: A fitted, scaled image
    """
    mean=scaler.mean_
    print(mean)
    var=scaler.scale_
    print(var)
    terrain_fit=np.zeros((int(len(y)/scaling),int(len(x)/scaling)))
    leny=int(len(y)/scaling)
    lenx=int(len(x)/scaling)
    print(lenx, leny)
    inverse_var=1/var
    for i in range(lenx):
        for j in range(leny):
            terrain_fit[j][i]=fit_func(beta,y[j*scaling],x[i*scaling],degree,mean,inverse_var)+mean_valz
    return terrain_fit
