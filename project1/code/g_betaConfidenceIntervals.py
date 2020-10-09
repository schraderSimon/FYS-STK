import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

"""make the data"""
terrain = imread("../data/Korea.tif") #Open image file
np.random.seed(sum([ord(c) for c in "CORONA"]))
datapoints= 50000 #number of sample points


x=np.random.randint(len(terrain),size=datapoints) #random integers
y=np.random.randint(len(terrain[1]),size=datapoints) #random integers for y
xy_array=np.column_stack((x,y))
z=[]
for xv,yv in xy_array:
    z.append(terrain[xv,yv])
z=np.array(z) #data to be fitted

"""Set initial parameters"""

nr_lambdas_ridge = 3

maxdeg=15
Z_table = 1.96
lambda_val_ridge = [0,0.01,1]

std_dev_ridge = np.zeros((maxdeg,len(lambda_val_ridge)))
max_beta = np.zeros((maxdeg,len(lambda_val_ridge)))
min_beta = np.zeros((maxdeg,len(lambda_val_ridge)))
counter = 0
for deg in range(1,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg) #create design matrix
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25) #split data
    z_train_scaled=z_train-np.mean(z_train) #subsract average
    z_test_scaled=z_test-np.mean(z_train)#subsract average
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    p = len(X[0,:])
    print('polydegree={0}'.format(int(deg)))

    for i in range(len(lambda_val_ridge)):
        beta_ridge, beta_var_ridge = RidgeRegression(X_train_scaled,z_train_scaled,lambda_val_ridge[i])

        z_diff = z_test_scaled-X_test_scaled @ beta_ridge
        sigma_squared_estimate = 1/(len(z_test_scaled)-p-1)*(z_diff @ z_diff)
        std_dev_ridge[counter,i] = round(np.sqrt(np.amax(beta_var_ridge*sigma_squared_estimate)),2)
        max_beta[counter,i] = np.round(np.amax(beta_ridge),2)
        min_beta[counter,i] = np.round(np.amin(beta_ridge),2)
    counter +=1

print('Random seed: '+'sum([ord(c) for c in "CORONA"]'+', datapoints = {0} ,Z_table = {2}, deg.max. = {1}, lambda_vals={3}'.format(datapoints,maxdeg,Z_table,lambda_val_ridge))



print('Max Std.dev. Ridge')
print(std_dev_ridge)

""" OUTPUT CSV """
dict = {'poly.deg.': [i for i in range(1,maxdeg+1)], 'OLS C.I.R.':  np.round(Z_table*std_dev_ridge[:,0],2), 'OLS_b_max':  max_beta[:,0], 'OLS_b_min':  min_beta[:,0],
'Ridge C.I.R. (L = {0})'.format(lambda_val_ridge[1]): np.round(Z_table*std_dev_ridge[:,1],2), 'Ridge_b_max(L = {0})'.format(lambda_val_ridge[1]):  max_beta[:,1],'Ridge_b_min(L = {0})'.format(lambda_val_ridge[1]):  min_beta[:,1],'Ridge C.I.R. (L = {0})'.format(lambda_val_ridge[2]): np.round(Z_table*std_dev_ridge[:,2],2), 'Ridge_b_max(L = {0})'.format(lambda_val_ridge[2]):  max_beta[:,2], 'Ridge_b_min(L = {0})'.format(lambda_val_ridge[2]):  min_beta[:,2]}
df = pd.DataFrame(dict)
df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/g_confidence_intervals_maxdeg({0}).csv'.format(maxdeg))
