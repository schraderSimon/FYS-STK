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
maxdeg=5

Lambda1 = 0.1
Lambda2 = 1

sigma=0.05
z=FrankeFunction(x,y)+np.random.normal(0,sigma, datapoints)

beta_variance=np.zeros((maxdeg,maxdeg))
Z_table = 1.96#[2.326,1.96,1.645,1.282,0.674] 98,95,90,80 and 50% respectively

std_dev_OLS = np.zeros(maxdeg)
std_dev_ridge_L01 = np.zeros(maxdeg)
std_dev_ridge_L10 = np.zeros(maxdeg)

it = 0
for deg in range(1,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg,False)

    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train-np.mean(z_train)
    z_test_scaled=z_test-np.mean(z_train)
    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    #beta,beta_variance[it,:] = LinearRegression(X_train_scaled,z_train_scaled)
    beta_OLS,beta_var_OLS= LinearRegression(X_train_scaled,z_train_scaled)
    beta_ridge1, beta_var_ridge1 = RidgeRegression(X_train,z_train,Lambda1)
    beta_ridge2, beta_var_ridge2 = RidgeRegression(X_train,z_train,Lambda2)
    std_dev_OLS[it] = np.amax(np.sqrt(beta_var_OLS*sigma**2))
    std_dev_ridge_L01[it] = np.amax(np.sqrt(beta_var_ridge1*sigma**2))
    std_dev_ridge_L10[it] = np.amax(np.sqrt(beta_var_ridge2*sigma**2))
    it +=1
print('Max Std.dev. OLS')
print(std_dev_OLS)
print('Max Std.dev. Ridge')
print(std_dev_ridge_L01)
print('Max Std.dev. Ridge2')
print(std_dev_ridge_L10)

dict = {'polynomial degree': list(range(1,maxdeg +1)),r'OLS': Z_table*std_dev_OLS, r'Ridge ($\lambda = 0.1$)': Z_table*std_dev_ridge_L01,r'Ridge ($\lambda = 1.0$)': Z_table*std_dev_ridge_L10}
df = pd.DataFrame(dict)
df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/deltaBeta_conf_int_95_OLS_vs_Ridge.csv')