from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import matplotlib.pyplot as plt
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
n=15 #Data points for x
m=15 #Data points for y
np.random.seed(sum([ord(c) for c in "CoRoNa"]))
datapoints=500
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
maxdeg=10
n_bootstraps=300
z=FrankeFunction(x,y)+0.1*np.random.normal(0, 1, datapoints)
print(len(z))
MSE_train=np.zeros(maxdeg)
MSE_test=np.zeros(maxdeg)
R2_train=np.zeros(maxdeg)
R2_test=np.zeros(maxdeg)
for deg in range(1,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg,False)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
    z_train_scaled=z_train-np.mean(z_train)
    z_test_scaled=z_test-np.mean(z_train)
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    beta, beta_variance = LinearRegression(X_train_scaled,z_train_scaled)
    z_train_scaled_fit=X_train_scaled@beta
    MSE_train[deg-1]+=(MSE(z_train_scaled,z_train_scaled_fit))
    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit=X_test_scaled @ beta
        MSE_test[deg-1]+=(MSE(z_test_scaled,z_test_scaled_fit))
    R2_train[deg-1]+=(R2(z_train_scaled,z_train_scaled_fit))
    R2_test[deg-1]+=(R2(z_test_scaled,z_test_scaled_fit))
    MSE_test[deg-1]/=n_bootstraps
    print(f"Degree: {deg}")
    print("MSE_train: %f" %(MSE(z_train_scaled,z_train_scaled_fit)))
    print("MSE_test : %f" %(MSE(z_test_scaled,z_test_scaled_fit)))
    print("R2_train: %f" %(R2(z_train_scaled,z_train_scaled_fit)))
    print("R2_test: %f" %(R2(z_test_scaled,z_test_scaled_fit)))
plt.plot(range(1,maxdeg+1),MSE_train,label="MSE_train")
plt.plot(range(1,maxdeg+1),MSE_test,label="MSE_test")
plt.legend()
resample(X,z)
plt.show()
