"""
In this file, I calculate the mean square error
for all three methods, and I use k-fold Cross validation as well as bootstrap.
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(sum([ord(c) for c in "CORONA"]))
datapoints=150 #Nice data for (100,6) and (500,10), 0.1 random, corona
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
n_bootstraps=1000
n_bootstraps_lasso=1000
sigma=0.3
z=FrankeFunction(x,y)+np.random.normal(0,sigma, datapoints)
nr_lambdas = 100
min_lambda = -5 #lamba
max_lambda = 5
mindeg = 1 #the minimal degree to be calculated. Usually 1
maxdeg=11 #The maximal degree to be calculated
lasso_tol=1e-4 #Convergence criteria for LASSO
lasso_iterations=1e7 #Convergence criteria for LASSO
k=5
lambda_val = np.logspace(min_lambda,max_lambda,nr_lambdas)
MSEkfoldLASSO = np.zeros(maxdeg-mindeg +1)
MSEkfoldOLS = np.zeros(maxdeg-mindeg +1)
MSEkfoldRIDGE = np.zeros(maxdeg-mindeg +1)
MSEBOOTLASSO = np.zeros(maxdeg-mindeg +1)
MSEBOOTRIDGE = np.zeros(maxdeg-mindeg +1)
MSEBOOTOLS = np.zeros(maxdeg-mindeg +1)
MSE_test_kfoldLASSO_lambda=np.zeros(nr_lambdas)
MSE_test_kfoldRidge_lambda=np.zeros(nr_lambdas)
R2BOOTLASSO = np.zeros(maxdeg-mindeg +1)
R2BOOTRIDGE = np.zeros(maxdeg-mindeg +1)
R2BOOTOLS = np.zeros(maxdeg-mindeg +1)
for deg in range(mindeg,maxdeg+1):
    """For each degree"""
    X=DesignMatrix_deg2(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25) #split data in train and test
    z_train_scaled=z_train-np.mean(z_train) #substract average from test data
    z_test_scaled=z_test-np.mean(z_train)
    scaler=StandardScaler() #scaler
    scaler.fit(X_train) #Fit data
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    z_test_scaled_fit_OLS=np.zeros((len(z_test),n_bootstraps)) #fitted data OLS
    z_test_scaled_fit_LASSO=np.zeros((len(z_test),n_bootstraps_lasso)) #fitted data LASSO
    z_test_scaled_fit_RIDGE=np.zeros((len(z_test),n_bootstraps)) #fitted data ridge
    for i in range(nr_lambdas):
        """Find the ideal lambda value using K-fold Cross validation"""
        MSE_test_kfoldRidge_lambda[i] = KCrossValRidgeMSE(X,z,k,lambda_val[i])
        MSE_test_kfoldLASSO_lambda[i] = KCrossValLASSOMSE(X,z,k,lambda_val[i],lasso_tol,lasso_iterations)
    ideal_lambda_LASSO=lambda_val[np.argmin(MSE_test_kfoldLASSO_lambda)]
    ideal_lambda_RIGDE=lambda_val[np.argmin(MSE_test_kfoldRidge_lambda)]
        """Do Bootstrap"""

    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit_OLS[:,i]=X_test_scaled @ beta

        beta, beta_variance=RidgeRegression(X_b,z_b,ideal_lambda_RIGDE)
        z_test_scaled_fit_RIDGE[:,i]=X_test_scaled @ beta
        if(i<n_bootstraps_lasso): #drop LASSO when done with LASSO
            beta= LASSORegression(X_b,z_b,ideal_lambda_LASSO,lasso_tol,lasso_iterations)
            z_test_scaled_fit_LASSO[:,i]=X_test_scaled @ beta
    MSEBOOTOLS[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit_OLS,n_bootstraps)
    MSEBOOTLASSO[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit_LASSO,n_bootstraps_lasso)
    MSEBOOTRIDGE[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit_RIDGE,n_bootstraps)
    MSEkfoldRIDGE[deg-mindeg] = KCrossValRidgeMSE(X,z,k,ideal_lambda_RIGDE)
    MSEkfoldLASSO[deg-mindeg] = KCrossValLASSOMSE(X,z,k,ideal_lambda_LASSO,lasso_tol,lasso_iterations)
    MSEkfoldOLS[deg-mindeg] = KCrossValOLSMSE(X,z,k)
    R2BOOTLASSO[deg-mindeg] = bootstrap_r2(z_test_scaled,z_test_scaled_fit_LASSO,n_bootstraps_lasso)
    R2BOOTRIDGE[deg-mindeg] = bootstrap_r2(z_test_scaled,z_test_scaled_fit_RIDGE,n_bootstraps)
    R2BOOTOLS[deg-mindeg] = bootstrap_r2(z_test_scaled,z_test_scaled_fit_OLS,n_bootstraps)


write_csv=True
if (write_csv):
    #OUTPUTS CSV FILE CONTAINING MSE OF KFOLD-RIDGE OVER A SPAN OF LAMBDA VALUES (SAMPLE TYPE 2)
    dict = {"mindeg": mindeg, 'maxdeg':maxdeg,'nr_lambdas':max_lambda, 'min_lambda':min_lambda,'max_lambda':nr_lambdas,'sigma':sigma,"k":k ,'datapoints':datapoints , 'n_bootstrap': n_bootstraps,'n_bootstraps_lasso': n_bootstraps}
    dict["R2_lasso"]=R2BOOTLASSO; dict["R2_ridge"]=R2BOOTRIDGE; dict["R2_OLS"]=R2BOOTOLS
    dict["MSEBOOTRIDGE"]=MSEBOOTRIDGE;dict["MSEBOOTLASSO"]=MSEBOOTLASSO;dict["MSEBOOTOLS"]=MSEBOOTOLS
    dict["MSEkfoldRIDGE"]=MSEkfoldRIDGE;dict["MSEBOOTLASSO"]=MSEkfoldLASSO;dict["MSEBOOTOLS"]=MSEkfoldOLS
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/MSE_data_Franke_small.csv')


fig, (ax0, ax1) = plt.subplots(ncols=2,figsize=(10, 5))
xticks=np.arange(0, maxdeg+1, step=1)
x_axis=range(1,maxdeg+1)
ax0.set_title(r"$\sigma$=%.3f, datapoints: %d, bootstrap: %d"%(sigma,datapoints,n_bootstraps))
ax0.set_xticks(xticks)
ax0.set_xlabel("Polynomial degree")
ax0.set_ylabel("MSE")
ax0.set_ylim(0,0.2)
ax0.plot(x_axis,MSEBOOTOLS,label="MSE_OLS_BOOTSTRAP")
ax0.plot(x_axis,MSEBOOTRIDGE,label="MSE_RIDGE_BOOTSTRAP")
ax0.plot(x_axis,MSEBOOTLASSO,label="MSE_LASSO_BOOTSTRAP")
ax1.set_title(r"$\sigma$=%.3f, datapoints: %d, k: %d"%(sigma,datapoints,k))
ax1.set_xticks(xticks)
ax1.set_xlabel("Polynomial degree")
ax1.set_ylabel("MSE")
ax1.set_ylim(0,0.2)
ax1.plot(x_axis,MSEkfoldOLS,label="MSE_OLS_CrossVal")
ax1.plot(x_axis,MSEkfoldRIDGE,label="MSE_Ridge_CrossVal")
ax1.plot(x_axis,MSEkfoldLASSO,label="MSE_LASSO_CrossVal")

ax0.legend()
ax1.legend()
plt.savefig("../figures/MSE_different_methods_Franke_small.pdf")
plt.show()

plt.title(r"$\sigma$=%.3f, datapoints: %d, bootstrap: %d"%(sigma,datapoints,n_bootstraps))
plt.xticks(xticks)
plt.xlabel("Polynomial degree")
plt.ylabel("R2-score")
plt.ylim(0,1)
plt.plot(x_axis,R2BOOTOLS,label="OLS")
plt.plot(x_axis,R2BOOTRIDGE,label="Ridge")
plt.plot(x_axis,R2BOOTLASSO,label="Lasso")
plt.legend()
plt.savefig("../figures/R2_bootstrap_different_methods_Franke_small.pdf")
plt.show()

"""
run as: python3 calculate_different_MSE.py
"""
