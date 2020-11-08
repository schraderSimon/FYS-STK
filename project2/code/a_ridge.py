import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread

from function_library import *
import time
import pandas as pd
import sys
import seaborn as sns
sns.set()

try:
    datapoints=int(sys.argv[1])
    degree=int(sys.argv[2])
    batchsize=int(sys.argv[3])
    num_etas=int(sys.argv[4])
    epochs=int(sys.argv[5])
except:
    datapoints=2000
    degree=20
    batchsize=16
    num_etas=11#array length for etas & t1_values
    epochs=1000

"""Set up data"""
np.random.seed(sum([ord(c) for c in "CORONA"]))
terrain = imread("../data/Korea.tif")
x=np.random.randint(len(terrain),size=datapoints) #random integers
y=np.random.randint(len(terrain[1]),size=datapoints) #random integers for y
xy_array=np.column_stack((x,y))
z=[]
for xv,yv in xy_array:
    z.append(terrain[xv,yv])
z=np.array(z) #data to be fitted
X=DesignMatrix_deg2(x,y,degree,False)
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25) #split in train and test data Design Matrix
z_train_scaled=z_train-np.mean(z_train) #remove mean
z_test_scaled=z_test-np.mean(z_train) #remove mean
scaler=StandardScaler()
scaler.fit(X_train) #Scale
X_train_scaled=scaler.transform(X_train) #scale train Design matrix
X_test_scaled=scaler.transform(X_test) #scale test Design matrix

lambda_val = np.logspace(-30,0,11) # lambda values
MSE_test_kfoldRidge_lambda = np.zeros(len(lambda_val))
for i in range(len(lambda_val)):
    """Find the optimal analytical Ridge Regression error"""
    MSE_test_kfoldRidge_lambda[i] = KCrossValRidgeMSE(X,z,5,lambda_val[i])
optimal_lambda=lambda_val[np.argmin(MSE_test_kfoldRidge_lambda)] #find ideal lambda value (analytically)
print("Ideal lambda is: %e" %optimal_lambda)


theta_exact_Ridge,unimportant=RidgeRegression(X_train_scaled,z_train_scaled,optimal_lambda,False) #calculate Ridge Regression theta
z_fit=X_test_scaled @ theta_exact_Ridge
MSE_Ridge=MSE(z_test_scaled,z_fit) #Calculate Ridge Regression MSE
MSE_Ridge_lambda_eta_simple=np.zeros((num_etas,len(lambda_val)))
MSE_Ridge_lambda_eta_RMSprop=np.zeros((num_etas,len(lambda_val)))
MSE_Ridge_lambda_eta_ADAM=np.zeros((num_etas,len(lambda_val)))
eta_simple=np.logspace(-7,3,num_etas)
sgd=SGD_Ridge(X_train_scaled,z_train_scaled,n_epochs=epochs,batchsize=batchsize,Lambda=1)
for i in range(num_etas):"""For each learing rate"""

    for j in range(len(lambda_val)):"""For each lambda rate"""
        """Calculate MSE using ADAM, RMSpropand simple MSE"""
        sgd.Lambda=lambda_val[j] #Update Lambda paramter
        theta_simple=sgd.simple_fit(eta=eta_simple[i]); sgd.reset()
        theta_RMSprop=sgd.RMSprop(eta=eta_simple[i]); sgd.reset()
        theta_adam=sgd.ADAM(eta=eta_simple[i]); sgd.reset()
        MSE_Ridge_lambda_eta_ADAM[i,j]=MSE(z_test_scaled,X_test_scaled@theta_adam)
        MSE_Ridge_lambda_eta_simple[i,j]=MSE(z_test_scaled,X_test_scaled@theta_simple)
        MSE_Ridge_lambda_eta_RMSprop[i,j]=MSE(z_test_scaled,X_test_scaled@theta_RMSprop)
MSE_Ridge_lambda_eta_ADAM/=MSE_Ridge
MSE_Ridge_lambda_eta_simple/=MSE_Ridge
MSE_Ridge_lambda_eta_RMSprop/=MSE_Ridge

"""In order for the plot to be properly scaled, all "horrendous" values are replaced by nan. This does not affect the good values we're looking for"""
for i in range(num_etas):
    for j in range(len(lambda_val)):
        if MSE_Ridge_lambda_eta_ADAM[i,j]>10:
            MSE_Ridge_lambda_eta_ADAM[i,j]=np.nan
        if MSE_Ridge_lambda_eta_simple[i,j]>10:
            MSE_Ridge_lambda_eta_simple[i,j]=np.nan
        if MSE_Ridge_lambda_eta_RMSprop[i,j]>10:
            MSE_Ridge_lambda_eta_RMSprop[i,j]=np.nan
plt.figure(figsize=(10,10))
ax1=plt.subplot(311)
ax2=plt.subplot(312)
ax3=plt.subplot(313)
ax1.set_title(r"Simple SGD")
ax2.set_title(r"RMSProp")
ax3.set_title(r"ADAM")
ax1.ticklabel_format(useOffset=False)
ax2.ticklabel_format(useOffset=False)
ax3.ticklabel_format(useOffset=False)

sns.heatmap(MSE_Ridge_lambda_eta_simple, xticklabels=lambda_val, yticklabels=eta_simple,annot=True, ax=ax1, cmap="viridis")
sns.heatmap(MSE_Ridge_lambda_eta_RMSprop,xticklabels=lambda_val, yticklabels=eta_simple, annot=True, ax=ax2, cmap="viridis")
sns.heatmap(MSE_Ridge_lambda_eta_ADAM,xticklabels=lambda_val, yticklabels=eta_simple, annot=True, ax=ax3, cmap="viridis")
ax1.set_ylabel(r"$\eta$");ax2.set_ylabel(r"$\eta$");ax3.set_ylabel(r"$\eta$");
ax1.set_xlabel(r"$\lambda$"); ax2.set_xlabel(r"$\lambda$"); ax3.set_xlabel(r"$\lambda$");
plt.tight_layout()
plt.savefig("../figures/Ridge_error_SGD.pdf")
plt.show()

"""
python3 a_ridge.py 2000 20 16 11 1000
"""
