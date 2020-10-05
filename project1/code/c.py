from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 

""" PARAMETERS START"""
np.random.seed(sum([ord(c) for c in "corona"]))

k = 3
n_bootstraps=5000
maxdeg=5
datapoints=1000

""" PARAMETERS END"""

x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
z=FrankeFunction(x,y)+0.1*np.random.normal(0,1, datapoints)

MSE_test_kfold = np.zeros(maxdeg)
MSE_test_boot = np.zeros(maxdeg)



for deg in range(1,maxdeg+1):
    #Setting up data
    X=DesignMatrix_deg2(x,y,deg,True)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train
    z_test_scaled=z_test

    X_train_scaled=X_train
    X_test_scaled=X_test
    #Calling OLS regression
    beta, beta_variance = LinearRegression(X_train_scaled,z_train_scaled)
    #Scaling
    z_train_scaled_fit=X_train_scaled@beta
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
    #Performing Bootstrap
    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test_boot[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    #calling MSE from OLS Kfold Cross validation 
    MSE_test_kfold[deg-1] = KCrossValMSE(X,z,k)


""" CSV OUTPUT"""
#dict = {'polynomial degree': list(range(1,maxdeg +1)),'MSE Bootstrap': MSE_test_boot, 'MSE Kfold-crossvalidation': MSE_test_kfold}
#df = pd.DataFrame(dict)
#df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/Polydeg_Kfold_error.csv')

print('Random seed: '+'sum([ord(c) for c in "CORONA"]'+', k = {0}, n_bootstraps = {1}, maxdeg = {2}, datapoints = {3}'.format(k, n_bootstraps, maxdeg, datapoints))


print('MSE_test_boot')
print(MSE_test_boot)
print('MSE_test_kfold')
print(MSE_test_kfold)



"""
RUN EXAMPLES:

###############################################################################################################################################
Random seed: sum([ord(c) for c in "CORONA"], k = 4, n_bootstraps = 5000, maxdeg = 5, datapoints = 100
MSE_test_boot
[0.03120927 0.03108642 0.02616129 0.03247089 0.03919162]
MSE_test_kfold
[0.02635136 0.02431384 0.01880363 0.01912559 0.00987466]
###############################################################################################################################################
Random seed: sum([ord(c) for c in "CORONA"], k = 3, n_bootstraps = 5000, maxdeg = 5, datapoints = 1000
MSE_test_boot
[0.03132126 0.02788963 0.01936197 0.0150457  0.01282081]
MSE_test_kfold
[0.03472219 0.02931845 0.01779531 0.01434882 0.01218985]
###############################################################################################################################################
"""