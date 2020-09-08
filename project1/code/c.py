from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 

np.random.seed(sum([ord(c) for c in "corona"]))
#np.random.seed(670)

k = 4
n_bootstraps=5000

maxdeg=5

datapoints=100
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
z=FrankeFunction(x,y)+0.1*np.random.normal(0,1, datapoints)

MSE_test_kfold = np.zeros(maxdeg)
MSE_test_boot = np.zeros(maxdeg)



for deg in range(1,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg,True)
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
    z_train_scaled_fit=X_train_scaled@beta
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test_boot[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    MSE_test_kfold[deg-1] = KCrossValMSE(X,z,k)

dict = {'polynomial degree': list(range(1,maxdeg +1)),'MSE Bootstrap': MSE_test_boot, 'MSE Kfold-crossvalidation': MSE_test_kfold}

df = pd.DataFrame(dict)

df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/RidgeRegression_Lambda_error.csv')

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