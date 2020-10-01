from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
#np.random.seed(sum([ord(c) for c in "corona"]))
#np.random.seed(670)
np.random.seed(sum([ord(c) for c in "CORONA"]))
method="RIDGE"#Method name, important for file saving & plotting
datapoints=500
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
n_bootstraps=2000
sigma=0.05
z=FrankeFunction(x,y)+np.random.normal(0,sigma, datapoints)
k = 4
nr_lambdas = 120
min_lambda = -6
max_lambda=6
lambda_val = np.logspace(min_lambda,max_lambda,nr_lambdas)

mindeg = 1  #The minimal degree to be calculated
maxdeg = 17  #The maximal degree to be calculated


MSE_test_kfoldRidge_lambda = np.zeros(nr_lambdas)

MSE_test_kfoldRidge = np.zeros(maxdeg-mindeg +1)
MSE_test_kfold = np.zeros(maxdeg-mindeg +1)
MSE_test_boot = np.zeros(maxdeg-mindeg +1)
MSE_train=np.zeros(maxdeg-mindeg +1)
MSE_test=np.zeros(maxdeg-mindeg +1)
bias=np.zeros(maxdeg-mindeg +1)
variance=np.zeros(maxdeg-mindeg +1)
R2_train=np.zeros(maxdeg-mindeg +1)
R2_test=np.zeros(maxdeg-mindeg +1)
for deg in range(mindeg,maxdeg+1):
    print(deg)
    X=DesignMatrix_deg2(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train-np.mean(z_train) #remove mean
    z_test_scaled=z_test-np.mean(z_train) #remove mean
    scaler=StandardScaler()
    scaler.fit(X_train) #Scale

    X_train_scaled=scaler.transform(X_train) #scale train Design matrix
    X_test_scaled=scaler.transform(X_test) #scale test Design matrix

    """
    use K-Fold Cross validation to find optimal Lambda
    """
    for i in range(nr_lambdas):
        MSE_test_kfoldRidge_lambda[i] = KCrossValRidgeMSE(X,z,k,lambda_val[i])
    optimal_lambda=lambda_val[np.argmin(MSE_test_kfoldRidge_lambda)]
    beta,beta_variance=RidgeRegression(X_train_scaled,z_train_scaled,optimal_lambda) #the beta values for Ridge Regression
    z_train_scaled_fit=X_train_scaled@beta
    """
    Use bootstrap to get proper estimates
    """
    MSE_train[deg-1]+=(MSE(z_train_scaled,z_train_scaled_fit))
    R2_train[deg-1]+=(R2(z_train_scaled,z_train_scaled_fit))
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))

    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance=RidgeRegression(X_b,z_b,optimal_lambda)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    bias[deg-1] = bootstrap_bias(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    variance[deg-1] = bootstrap_variance(z_test_scaled,z_test_scaled_fit,n_bootstraps)
MSE_test_kfoldRidge_lambda_smooth = ArraySmoother(MSE_test_kfoldRidge_lambda,10)
plt.plot(np.log10(lambda_val),(MSE_test_kfoldRidge_lambda),'r--')
plt.plot(np.log10(lambda_val),(MSE_test_kfoldRidge_lambda_smooth))
plt.xlabel(r"$log_{10}(\lambda)$")
plt.ylabel(r"MSE")
plt.show()


plt.xticks(np.arange(0, maxdeg+1, step=1))
plt.title(r"Method: %s, $\sigma$=%.3f, datapoints: %d, bootstrap: %d"%(method,sigma,datapoints,n_bootstraps))

plt.plot(range(1,maxdeg+1),MSE_train,label="MSE_train")
plt.plot(range(1,maxdeg+1),MSE_test,label="MSE_test")
plt.plot(range(1,maxdeg+1),variance,label="variance")
plt.plot(range(1,maxdeg+1),bias,label=r"$bias^2$")
plt.xlabel("Polynomial degree")
plt.legend()
plt.savefig("../figures/Bias_Variance_%s.pdf"%(method))
write_csv=True
if (write_csv):
    #OUTPUTS CSV FILE CONTAINING MSE OF KFOLD-RIDGE OVER A SPAN OF LAMBDA VALUES (SAMPLE TYPE 2)
    dict = {'nr_lambdas':max_lambda, 'min_lambda':min_lambda,'max_lambda':nr_lambdas,'sigma':sigma ,'datapoints':datapoints , 'n_bootstrap': n_bootstraps, 'train_MSE':MSE_train, 'test_MSE': MSE_test, 'bias':bias, 'variance':variance}
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/Ridge_data.csv')
plt.show()
