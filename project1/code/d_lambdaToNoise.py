from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 



k = 4

nr_lambdas = 300
min_lambda = -3
lambda_val = np.logspace(min_lambda,7,nr_lambdas)

deg = 6

datapoints=90

x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)

nr_noiseVal = 200
function_noise = np.linspace(0,1,nr_noiseVal)



MSE_test_kfoldRidge_lambda = np.zeros((nr_lambdas,nr_noiseVal))

X=DesignMatrix_deg2(x,y,deg,True)

t = 0
for noise in function_noise:
    z=FrankeFunction(x,y)+noise*np.random.normal(0,1, datapoints)
    for i in range(nr_lambdas):
        #Averaging over two runs to hopefully avoid extreme outliers
        MSE_test_kfoldRidge_lambda[i,t] = (KCrossValRidgeMSE(X,z,k,lambda_val[i]) + KCrossValRidgeMSE(X,z,k,lambda_val[i]))/2
    t += 1
    print(t, "out of ",nr_noiseVal)

min_lambda = np.zeros(nr_noiseVal)


for i in range(len(min_lambda)):
    min_lambda[i] = min(MSE_test_kfoldRidge_lambda[:,i])

plt.plot(function_noise,np.log10(min_lambda))
plt.xlabel("Function noise")
plt.ylabel("Log10(Lambda) of minimum MSE")
plt.title("The lambda value resulting in the smalles MSE as a function of the noise coefficient")
plt.show()



