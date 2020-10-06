from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

""" PARAMETERS START"""
np.random.seed(sum([ord(c) for c in "corona"]))
k = 3
n_bootstraps=5000
datapoints=1400
nr_lambdas = 1
min_lambda = -2
max_lambda = -2
mindeg = 4
maxdeg = 8

#set nr_lambdas to 1 and lambda_val interval to desired Lambda for best result
csv_polydegree_comp = False
#set mindeg, maxdeg to a zero degree difference at the polydegree of choice for best output
csv_Lambdaval_comp = False
""" PARAMETERS END"""

#Creating a range of Lambda values
lambda_val = np.logspace(min_lambda,max_lambda,nr_lambdas)

#Setting up data
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
z=FrankeFunction(x,y)+np.random.normal(0,0.3, datapoints)

#Creating empty arrays for outputs
MSE_test_kfoldRidge_lambda = np.zeros(nr_lambdas)

MSE_test_kfoldRidge = np.zeros(maxdeg-mindeg +1)
MSE_test_kfold = np.zeros(maxdeg-mindeg +1)
MSE_test_boot = np.zeros(maxdeg-mindeg +1)


for deg in range(mindeg,maxdeg+1):
    #Data setup
    X=DesignMatrix_deg2(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train-np.mean(z_train)
    z_test_scaled=z_test-np.mean(z_train)

    X_train_scaled=X_train
    X_test_scaled=X_test

    beta, beta_variance = LinearRegression(X_train_scaled,z_train_scaled)
    z_train_scaled_fit=X_train_scaled @ beta
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
    #Bootstrap 
    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test_boot[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    #Calling kfold functions
    MSE_test_kfold[deg-mindeg] = KCrossValMSE(X,z,k)
    MSE_test_kfoldRidge[deg-mindeg] = KCrossValRidgeMSE(X,z,k,min_lambda)
    for i in range(nr_lambdas):
        MSE_test_kfoldRidge_lambda[i] = KCrossValRidgeMSE(X,z,k,lambda_val[i])


""" CSV OUTPUT"""
if (csv_polydegree_comp):
    #OUTPUTS CSV FILE CONTAINING MSE COMPARISONS BETWEEN BOOTSTRAP, KFOLD-OLS AND KFOLD-RIDGE OVER A SPAN OF POLYNOMIAL DEGREES (SAMPLE TYPE 1)
    dict = {'polynomial degree': list(range(mindeg,maxdeg +1)),'MSE Bootstrap': MSE_test_boot, 'MSE Kfold-crossvalidation': MSE_test_kfold,'MSE Kfold-crossvalidation Ridge': MSE_test_kfoldRidge}
    df = pd.DataFrame(dict)
    df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/KfoldRidge_polydegree_comparison_error.csv')

if (csv_Lambdaval_comp):
    #OUTPUTS CSV FILE CONTAINING MSE OF KFOLD-RIDGE OVER A SPAN OF LAMBDA VALUES (SAMPLE TYPE 2)
    dict = {'Lambda value': lambda_val,'MSE Kfold-crossvalidation Ridge': MSE_test_kfoldRidge_lambda}
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/RidgeRegression_Lambda_error.csv')
    plt.plot(np.log10(lambda_val),MSE_test_kfoldRidge_lambda)
    plt.show()

print('Random seed: '+'sum([ord(c) for c in "corona"]'+', k = {0}, n_bootstraps= {1}, datapoints= {2}, nr_lambdas = {3} , min_lambda = {4} , max_lambda = {5} , mindeg = {6} , maxdeg = {7} '.format(k, n_bootstraps, datapoints, nr_lambdas, min_lambda, max_lambda, mindeg, maxdeg))


print('MSE_test_boot')
print(MSE_test_boot)
print('MSE_test_kfold')
print(MSE_test_kfold)
print('MSE_test_kfoldRidge')
print(MSE_test_kfoldRidge)



"""
RUN EXAMPLES:

###############################################################################################################################################
Random seed: sum([ord(c) for c in "corona"], k = 4, n_bootstraps= 3000, datapoints= 800, nr_lambdas = 1 , min_lambda = -3 , max_lambda = -3 , mindeg = 1 , maxdeg = 6
MSE_test_boot
[0.1640776  0.1466464  0.10042615 0.08909936 0.10280794 0.12322569]
MSE_test_kfold
[0.13941575 0.1153697  0.09951944 0.10114366 0.1001922  0.1005203 ]
MSE_test_kfoldRidge
[0.11541569 0.10842707 0.15454377 1.11890843 0.1060322  7.92230779]
###############################################################################################################################################
Random seed: sum([ord(c) for c in "corona"], k = 3, n_bootstraps= 5000, datapoints= 1400, nr_lambdas = 1 , min_lambda = -2 , max_lambda = -2 , mindeg = 4 , maxdeg = 8 
MSE_test_boot
[0.08536207 0.09197229 0.09700139 0.0999154  0.09688921]
MSE_test_kfold
[0.09780281 0.09582772 0.09300836 0.09357938 0.09514385]
MSE_test_kfoldRidge
[ 0.10617896  0.1547338   0.11203348  0.15916627 50.87442912]
###############################################################################################################################################
"""


