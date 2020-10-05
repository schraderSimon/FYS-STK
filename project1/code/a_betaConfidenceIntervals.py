from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import matplotlib.pyplot as plt
import pandas as pd

""" PARAMETERS START"""
np.random.seed(sum([ord(c) for c in "CORONA"]))

datapoints=500 #Nice data for (100,6) and (500,10), 0.1 random, corona

Z_table = 1.96#[2.326,1.96,1.645,1.282,0.674] 98,95,90,80 and 50% respectively

maxdeg=8

Lambda1 = 0.01
Lambda2 = 10

sigma=0.10
""" PARAMETERS END"""

method="OLS"

x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)

z=FrankeFunction(x,y)+np.random.normal(0,sigma, datapoints)

beta_variance=np.zeros((maxdeg,maxdeg))

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
    

    beta_OLS,beta_var_OLS= LinearRegression(X_train_scaled,z_train_scaled)
    beta_ridge1, beta_var_ridge1 = RidgeRegression(X_train,z_train,Lambda1)
    beta_ridge2, beta_var_ridge2 = RidgeRegression(X_train,z_train,Lambda2)
    std_dev_OLS[it] = np.amax(np.sqrt(beta_var_OLS*sigma**2))
    std_dev_ridge_L01[it] = np.amax(np.sqrt(beta_var_ridge1*sigma**2))
    std_dev_ridge_L10[it] = np.amax(np.sqrt(beta_var_ridge2*sigma**2))
    it +=1

print('Random seed: '+'sum([ord(c) for c in "CORONA"]'+', datapoints = {0} ,Z_table = {1}, maxdeg = {2}, Lambda1 = {3}, Lambda2 = {4}, sigma = {5}'.format(datapoints,Z_table,maxdeg,Lambda1,Lambda2,sigma))


print('Max Std.dev. OLS')
print(std_dev_OLS)
print('Max Std.dev. Ridge')
print(std_dev_ridge_L01)
print('Max Std.dev. Ridge2')
print(std_dev_ridge_L10)

""" OUTPUT CSV """
#dict = {'polynomial degree': list(range(1,maxdeg +1)),'OLS': "%f.2"Z_table*std_dev_OLS, 'Ridge (lambda = 0.1)': Z_table*std_dev_ridge_L01,'Ridge (lambda = 1.0)': Z_table*std_dev_ridge_L10}
#df = pd.DataFrame(dict)
#df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/deltaBeta_conf_int_95_OLS_vs_Ridge.csv')

"""
RUN EXAMPLES:

###############################################################################################################################################
Random seed: sum([ord(c) for c in "CORONA"], datapoints = 500 ,Z_table = 1.96, maxdeg = 5, Lambda1 = 0.1, Lambda2 = 1, sigma = 0.05
Max Std.dev. OLS
[0.00258586 0.01150159 0.06512398 0.32244928 1.51893295]
Max Std.dev. Ridge
[0.00692546 0.03137258 0.10554931 0.12750543 0.14184825]
Max Std.dev. Ridge2
[0.00683749 0.0241607  0.039384   0.04373097 0.0464925 ]
###############################################################################################################################################
Random seed: sum([ord(c) for c in "CORONA"], datapoints = 500 ,Z_table = 1.96, maxdeg = 8, Lambda1 = 0.01, Lambda2 = 10, sigma = 0.1
Max Std.dev. OLS
[5.17172648e-03 2.30031815e-02 1.30247969e-01 6.44898554e-01
 3.03786590e+00 1.80433924e+01 9.45097466e+01 5.17479422e+02]
Max Std.dev. Ridge
[0.01386893 0.06520251 0.35817165 0.72727571 0.81848178 0.88287024
 0.91850855 0.9348185 ]
Max Std.dev. Ridge2
[0.01224847 0.02541776 0.02818247 0.02948129 0.03018964 0.03065545
 0.03084895 0.03098503]
###############################################################################################################################################
"""