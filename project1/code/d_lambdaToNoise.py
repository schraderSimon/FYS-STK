from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 


""" PARAMETERS START"""
k = 4
nr_lambdas = 80
smallest_lambda = -3
deg = 4
datapoints=200
nr_noiseVal = 40
""" PARAMETERS END"""


#Creating a list of values for function noise

function_noise = np.linspace(0,1,nr_noiseVal)

lambda_val = np.logspace(smallest_lambda,1,nr_lambdas)

x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)

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
plt.xlabel(r"Function noise ($\sigma$)")
plt.ylabel('$\log_{10}(\lambda)$'+' of minimum MSE')
plt.title(r"$\lambda$ resulting in smallest MSE for each $\sigma$ (poly. deg.= " +str(deg)+r', #datapoints = '+str(datapoints) +r' )' )
plt.show()

print('Random seed: '+'sum([ord(c) for c in "corona"]' + ',k = {0}, nr_lambdas = {1}, smallest_lambda = {2}, deg = {3}, datapoints = {4}, nr_noiseVal = {5}'.format(k , nr_lambdas  , smallest_lambda  , deg  , datapoints , nr_noiseVal ))

print('Lambda resulting in smallest MSE')
print(min_lambda)


"""
RUN EXAMPLES:

###############################################################################################################################################
Random seed: sum([ord(c) for c in "corona"],k = 4, nr_lambdas = 80, smallest_lambda = -3, deg = 4, datapoints = 200, nr_noiseVal = 40
Lambda resulting in smallest MSE
[0.00480359 0.0055031  0.00801271 0.01237581 0.01625353 0.02241479
 0.03373827 0.03653577 0.04834668 0.05958756 0.07687711 0.09220621
 0.10292206 0.11424014 0.127878   0.17096495 0.18752849 0.23711585
 0.23612681 0.26127658 0.25646683 0.33055715 0.32883442 0.41701454
 0.40256273 0.35451488 0.54922584 0.41894652 0.64126143 0.71786912
 0.56203241 0.74364191 0.76392082 0.60484718 0.83028891 0.8694009
 0.85052272 0.73369273 0.93014298 1.01511184]
###############################################################################################################################################

"""


