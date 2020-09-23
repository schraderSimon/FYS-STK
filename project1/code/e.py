from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
#np.random.seed(sum([ord(c) for c in "corona"]))
#np.random.seed(670)
np.random.seed(sum([ord(c) for c in "CORONA"]))
method="LASSO"
datapoints=500 #Nice data for (100,6) and (500,10), 0.1 random, corona
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
maxdeg=10
n_bootstraps=5000
sigma=0.05
z=FrankeFunction(x,y)+np.random.normal(0,sigma, datapoints)

k = 4
nr_lambdas = 200
min_lambda = -6
max_lambda=6
lambda_val = np.logspace(min_lambda,max_lambda,nr_lambdas)

mindeg = 1
maxdeg = 10


MSE_test_kfoldLASSO_lambda = np.zeros(nr_lambdas)

MSE_test_kfoldLASSO = np.zeros(maxdeg-mindeg +1)
MSE_test_kfold = np.zeros(maxdeg-mindeg +1)
MSE_test_boot = np.zeros(maxdeg-mindeg +1)
MSE_train=np.zeros(maxdeg-mindeg +1)
MSE_test=np.zeros(maxdeg-mindeg +1)
bias=np.zeros(maxdeg-mindeg +1)
variance=np.zeros(maxdeg-mindeg +1)
R2_train=np.zeros(maxdeg-mindeg +1)
R2_test=np.zeros(maxdeg-mindeg +1)
for deg in range(mindeg,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train-np.mean(z_train)
    z_test_scaled=z_test-np.mean(z_train)
    scaler=StandardScaler()
    scaler.fit(X_train)
    #X_train_scaled=X_train
    #X_test_scaled=X_test
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    #MSE_test_kfoldLASSO[deg-mindeg] = KCrossValLASSOMSE(X,z,k,min_lambda)

    """
    use K-Fold Cross validation to find optimal Lambda
    """
    for i in range(nr_lambdas):
        MSE_test_kfoldLASSO_lambda[i] = KCrossValLASSOMSE(X,z,k,lambda_val[i])
    optimal_lambda=lambda_val[np.argmin(MSE_test_kfoldLASSO_lambda)]
    z_train_scaled_fit=X_train_scaled@LASSORegression(X_train_scaled,z_train_scaled,optimal_lambda)
    """
    Use bootstrap to get proper estimates
    """
    MSE_train[deg-1]+=(MSE(z_train_scaled,z_train_scaled_fit))
    R2_train[deg-1]+=(R2(z_train_scaled,z_train_scaled_fit))
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))

    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta= LASSORegression(X_b,z_b,optimal_lambda)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    bias[deg-1] = bootstrap_bias(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    variance[deg-1] = bootstrap_variance(z_test_scaled,z_test_scaled_fit,n_bootstraps)
MSE_test_kfoldLASSO_lambda_smooth = ArraySmoother(MSE_test_kfoldLASSO_lambda,10)
plt.plot(np.log10(lambda_val),(MSE_test_kfoldLASSO_lambda),'r--')
plt.plot(np.log10(lambda_val),(MSE_test_kfoldLASSO_lambda_smooth))
plt.xlabel(r"$log_{10}(\lambda)$")
plt.ylabel(r"MSE")
plt.show()
plt.xticks(np.arange(0, maxdeg+1, step=1))
plt.title(r"Method: %s, $\sigma$=%.3f, datapoints: %d, bootstrap: %d"%(method,sigma,datapoints,n_bootstraps))

plt.plot(range(1,maxdeg+1),MSE_train,label="MSE_train")
plt.plot(range(1,maxdeg+1),MSE_test,label="MSE_test")
plt.plot(range(1,maxdeg+1),variance,label="variance")
plt.plot(range(1,maxdeg+1),bias,label=r"$bias^2$")
plt.savefig("../figures/Bias_Variance_%s_.png"%(method))
plt.xlabel("Polynomial degree")
plt.legend()
plt.show()
