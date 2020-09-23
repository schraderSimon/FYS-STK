from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import matplotlib.pyplot as plt

#np.random.seed(sum([ord(c) for c in "corona"]))
datapoints=300 #Nice data for (100,6) and (500,10), 0.1 random, corona
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
maxdeg=9
n_bootstraps=5000
z=FrankeFunction(x,y)+np.random.normal(0,0.1, datapoints)
MSE_train=np.zeros(maxdeg)
MSE_test=np.zeros(maxdeg)
bias=np.zeros(maxdeg)
variance=np.zeros(maxdeg)
R2_train=np.zeros(maxdeg)
R2_test=np.zeros(maxdeg)
for deg in range(1,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg,False)
    #print(X)
    #X_shuffled=X.copy()
    #np.random.shuffle(X_shuffled)
    #print(X_shuffled)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train-np.mean(z_train)
    z_test_scaled=z_test-np.mean(z_train)
    scaler=StandardScaler()
    scaler.fit(X_train)
    #X_train_scaled=X_train
    #X_test_scaled=X_test
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    beta, beta_variance = LinearRegression(X_train_scaled,z_train_scaled)
    z_train_scaled_fit=X_train_scaled@beta
    MSE_train[deg-1]+=(MSE(z_train_scaled,z_train_scaled_fit))
    R2_train[deg-1]+=(R2(z_train_scaled,z_train_scaled_fit))
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    bias[deg-1] = bootstrap_bias(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    variance[deg-1] = bootstrap_variance(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    print(f"Degree: {deg}")

plt.xticks(np.arange(0, maxdeg+1, step=1))
plt.plot(range(1,maxdeg+1),MSE_train,label="MSE_train")
plt.plot(range(1,maxdeg+1),MSE_test,label="MSE_test")
plt.plot(range(1,maxdeg+1),variance,label="variance")
plt.plot(range(1,maxdeg+1),bias,label=r"$bias^2$")
plt.xlabel("Polynomial degree")
plt.legend()
resample(X,z)
plt.show()
