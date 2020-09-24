from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(sum([ord(c) for c in "CORONA"]))
method="OLS"
datapoints=500 #Nice data for (100,6) and (500,10), 0.1 random, corona
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
maxdeg=10
n_bootstraps=5000
sigma=0.05
z=FrankeFunction(x,y)+np.random.normal(0,sigma, datapoints)
MSE_train=np.zeros(maxdeg)
MSE_test=np.zeros(maxdeg)
bias=np.zeros(maxdeg)
variance=np.zeros(maxdeg)
R2_train=np.zeros(maxdeg)
R2_test=np.zeros(maxdeg)
for deg in range(1,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg,False)

    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train-np.mean(z_train)
    z_test_scaled=z_test-np.mean(z_train)
    scaler=StandardScaler()
    scaler.fit(X_train)
    #X_train_scaled=X_train
    #X_test_scaled=X_test
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    beta = OLS_SVD(X_train_scaled,z_train_scaled)
    z_train_scaled_fit=X_train_scaled@beta
    MSE_train[deg-1]+=(MSE(z_train_scaled,z_train_scaled_fit))
    R2_train[deg-1]+=(R2(z_train_scaled,z_train_scaled_fit))
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta = OLS_SVD(X_b,z_b)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    bias[deg-1] = bootstrap_bias(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    variance[deg-1] = bootstrap_variance(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    print(f"Degree: {deg}")

plt.xticks(np.arange(0, maxdeg+1, step=1))
plt.title(r"Method: %s, $\sigma$=%.3f, datapoints: %d, bootstrap: %d"%(method,sigma,datapoints,n_bootstraps))
plt.plot(range(1,maxdeg+1),MSE_train,label="MSE_train")
plt.plot(range(1,maxdeg+1),MSE_test,label="MSE_test")
plt.plot(range(1,maxdeg+1),variance,label="variance")
plt.plot(range(1,maxdeg+1),bias,label=r"$bias^2$")
plt.xlabel("Polynomial degree")
plt.savefig("../figures/Bias_Variance_%s_.pdf"%(method))
plt.legend()
write_csv=True
if (write_csv):
    #OUTPUTS CSV FILE CONTAINING MSE OF KFOLD-RIDGE OVER A SPAN OF LAMBDA VALUES (SAMPLE TYPE 2)
    dict = {'sigma':sigma ,'datapoints':datapoints , 'n_bootstrap': n_bootstraps, 'train_MSE':MSE_train, 'test_MSE': MSE_test, 'bias':bias, 'variance':variance}
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/OLS_data.csv')

plt.show()
