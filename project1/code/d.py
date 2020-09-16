from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(sum([ord(c) for c in "corona"]))
#np.random.seed(670)

k = 4
n_bootstraps=5000

nr_lambdas = 800
min_lambda = -5
max_lambda = 5
lambda_val = np.logspace(min_lambda,max_lambda,nr_lambdas)

mindeg = 6
maxdeg = 6

#set nr_lambdas to 1 and lambda_val interval to desired Lambda for best result
csv_polydegree_comp = False
#set mindeg, maxdeg to a zero degree difference at the polydegree of choice for best output
csv_Lambdaval_comp = True


datapoints=2000
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
z=FrankeFunction(x,y)+np.random.normal(0,1, datapoints)

MSE_test_kfoldRidge_lambda = np.zeros(nr_lambdas)

MSE_test_kfoldRidge = np.zeros(maxdeg-mindeg +1)
MSE_test_kfold = np.zeros(maxdeg-mindeg +1)
MSE_test_boot = np.zeros(maxdeg-mindeg +1)


for deg in range(mindeg,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train#-np.mean(z_train)
    z_test_scaled=z_test#-np.mean(z_train)
    #scaler=StandardScaler()
    #scaler.fit(X_train)
    X_train_scaled=X_train
    X_test_scaled=X_test
    #X_train_scaled=scaler.transform(X_train)
    #X_test_scaled=scaler.transform(X_test)
    beta, beta_variance = LinearRegression(X_train_scaled,z_train_scaled)
    z_train_scaled_fit=X_train_scaled @ beta
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
    for i in range(1):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test_boot[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    MSE_test_kfold[deg-mindeg] = KCrossValMSE(X,z,k)
    MSE_test_kfoldRidge[deg-mindeg] = KCrossValRidgeMSE(X,z,k,min_lambda)
    for i in range(nr_lambdas):
        MSE_test_kfoldRidge_lambda[i] = KCrossValRidgeMSE(X,z,k,lambda_val[i])


if (csv_polydegree_comp):
    #OUTPUTS CSV FILE CONTAINING MSE COMPARISONS BETWEEN BOOTSTRAP, KFOLD-OLS AND KFOLD-RIDGE OVER A SPAN OF POLYNOMIAL DEGREES (SAMPLE TYPE 1)
    dict = {'polynomial degree': list(range(mindeg,maxdeg +1)),'MSE Bootstrap': MSE_test_boot, 'MSE Kfold-crossvalidation': MSE_test_kfold,'MSE Kfold-crossvalidation Ridge': MSE_test_kfoldRidge}
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/KfoldRidge_polydegree_comparison_error.csv')

if (csv_Lambdaval_comp):
    #OUTPUTS CSV FILE CONTAINING MSE OF KFOLD-RIDGE OVER A SPAN OF LAMBDA VALUES (SAMPLE TYPE 2)
    dict = {'Lambda value': lambda_val,'MSE Kfold-crossvalidation Ridge': MSE_test_kfoldRidge_lambda}
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/RidgeRegression_Lambda_error.csv')
    MSE_test_kfoldRidge_lambda_smooth = ArraySmoother(MSE_test_kfoldRidge_lambda,10)
    plt.plot(np.log10(lambda_val),MSE_test_kfoldRidge_lambda,'r--')
    plt.plot(np.log10(lambda_val),MSE_test_kfoldRidge_lambda_smooth)
    plt.show()




"""
plt.xticks(np.arange(0, maxdeg+1, step=1))
plt.plot(range(1,maxdeg+1),MSE_train,label="MSE_train")
plt.plot(range(1,maxdeg+1),MSE_test,label="MSE_test")
plt.plot(range(1,maxdeg+1),variance,label="variance")
plt.plot(range(1,maxdeg+1),bias,label="bias")
plt.legend()
resample(X,z)
plt.show()
"""
