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

    #The value of p from equation 4 in the report is calculated by the number of degrees of freedom in the model
    #namely the coefficients in the polynomial
    p = len(X[0,:])
    print('polydegree={0}'.format(int(deg)))

    for i in range(len(lambda_val_ridge)):
        beta_ridge, beta_var_ridge = RidgeRegression(X_train_scaled,z_train_scaled,lambda_val_ridge[i])

        #calculating the difference between the model estimates and the function values
        z_diff = z_test_scaled-X_test_scaled @ beta_ridge
        #Calculating the expression 4 in the report
        sigma_squared_estimate = 1/(len(z_test_scaled)-p-1)*(z_diff @ z_diff)
        std_dev_ridge[counter,i] = round(np.sqrt(np.amax(beta_var_ridge*sigma_squared_estimate)),2)
        max_beta[counter,i] = np.round(np.amax(beta_ridge),2)
        min_beta[counter,i] = np.round(np.amin(beta_ridge),2)
    counter +=1

print('Random seed: '+'sum([ord(c) for c in "CORONA"]'+', datapoints = {0} ,Z_table = {2}, deg.max. = {1}, lambda_vals={3}'.format(datapoints,maxdeg,Z_table,lambda_val_ridge))



print('Max Std.dev. Ridge')
print(std_dev_ridge)



""" OUTPUT CSV """

dict = {'p.d.': [i for i in range(1,maxdeg+1)], 'OLS C.I.':  ["%.2E" % number for number in np.round(Z_table*std_dev_ridge[:,0],2)], 'OLS b_{max}':  ["%.2E" % number for number in max_beta[:,0]], 'OLS b_{min}': ["%.2E" % number for number in min_beta[:,0]], 'R({0}) C.I.'.format(lambda_val_ridge[1]): ["%.2E" % number for number in np.round(Z_table*std_dev_ridge[:,1],2)], 'R. b_{max}':  ["%.2E" % number for number in max_beta[:,1]],'R. b_{min}':  ["%.2E" % number for number in min_beta[:,1]],'R.({0}) C.I.'.format(lambda_val_ridge[2]): ["%.2E" % number for number in np.round(Z_table*std_dev_ridge[:,2],2)], 'R._2 b_{max}':  ["%.2E" % number for number in max_beta[:,2]], 'R._2 b_{min}':  ["%.2E" % number for number in min_beta[:,2]]}
df = pd.DataFrame(dict)
df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/g_confidence_intervals_maxdeg_{0}.csv'.format(maxdeg),index=False)

"""
Run Example:
Random seed: sum([ord(c) for c in "CORONA"], datapoints = 50000 ,Z_table = 1.96, deg.max. = 15, lambda_vals=[0, 0.01, 1]
Max Std.dev. Ridge
[[1.51000000e+00 1.51000000e+00 1.51000000e+00]
 [5.31000000e+00 5.31000000e+00 5.31000000e+00]
 [2.60500000e+01 2.60400000e+01 2.57200000e+01]
 [1.28060000e+02 1.27600000e+02 9.78000000e+01]
 [5.40600000e+02 4.91910000e+02 1.28130000e+02]
 [2.89653000e+03 1.11619000e+03 1.36270000e+02]
 [1.47870900e+04 1.19276000e+03 1.45280000e+02]
 [7.42867000e+04 1.26451000e+03 1.54830000e+02]
 [3.97810430e+05 1.35832000e+03 1.58080000e+02]
 [2.12928902e+06 1.41993000e+03 1.58870000e+02]
 [1.06439332e+07 1.46686000e+03 1.59540000e+02]
 [5.35286089e+07 1.51121000e+03 1.62070000e+02]
 [2.49256454e+07 1.52466000e+03 1.61760000e+02]
 [1.29024831e+07 1.53464000e+03 1.62110000e+02]
 [3.59159606e+07 1.55956000e+03 1.64030000e+02]]
"""