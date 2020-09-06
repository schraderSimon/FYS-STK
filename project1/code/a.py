from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from random import random, seed
from help_functions import *
import sys
n=20 #Data points for x
m=20 #Data points for y
maxdeg=5
x=np.linspace(0,1,n).reshape(-1, 1)
y=np.linspace(0,1,m).reshape(-1, 1)
x,y= np.meshgrid(x,y)
z=np.ravel(FrankeFunction(x,y))
print(z)
print(x)
Intercept_include=False
addor=0
p=round((maxdeg+1)*(maxdeg+2)/2)-1
if Intercept_include:
    p+=1
    addor=1
X=np.zeros((m*n,p))
if Intercept_include:
    X[:,0]=1

X[:,0+addor]=np.ravel(x) # Adds x on the first column
X[:,1+addor]=np.ravel(y) # Adds y on the second column

"""
Create the design matrix:
X[:,3] is x**2 * y**0, X[:,4] is x**1 * y **1,
X[:,7] is x**2 * y ** 1 and so on. Up to degree maxdeg
"""
count=2+addor
for i in range(2,maxdeg+1):
    for j in range(i+1):
        X[:,count]=X[:,0+addor]**j*X[:,1+addor]**(i-j)
        count+=1;


X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
z_train_scaled=z_train-np.mean(z_train)
z_test_scaled=z_test-np.mean(z_train)
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
for i in range(p):
    print(i)
    print(X_test_scaled[:,i])
try:
    np.linalg.inv(X_train.T @ X_train)
except:
    print("X_train.T @ X_train is not invertible")
try:
    np.linalg.inv(X.T @ X)
except:
    print("X.T @ X is not invertible")
try:
    np.linalg.inv(X_train_scaled.T @ X_train_scaled)
except:
    print("X_train_scaled.T @ X_train_scaled is not invertible")
beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train_scaled
