from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 


""" PARAMETERS START"""
np.random.seed(sum([ord(c) for c in "corona"]))
k = 7
nr_lambdas = 100
smallest_lambda = -3
polyMax = 7
datapoints=1000
""" PARAMETERS END"""

lambda_val = np.logspace(smallest_lambda,1,nr_lambdas)
deg = [i for i in range(1,polyMax+1)]
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)

MSE_test_kfoldRidge_lambda = np.zeros((nr_lambdas,polyMax))



t = 0
for p in deg:
    X=DesignMatrix_deg2(x,y,p,True)
    z=FrankeFunction(x,y)+0.2*np.random.normal(0,1, datapoints)
    for i in range(nr_lambdas):
        #Averaging over two runs to hopefully avoid extreme outliers
        MSE_test_kfoldRidge_lambda[i,t] = (KCrossValRidgeMSE(X,z,k,lambda_val[i]) + KCrossValRidgeMSE(X,z,k,lambda_val[i]))/2
        print(i,'out of ',nr_lambdas)
    t += 1
    print(t, "out of ",polyMax)

min_lambda = np.zeros(polyMax)


for i in range(len(min_lambda)):
    min_lambda[i] = min(MSE_test_kfoldRidge_lambda[:,i])

plt.plot(deg,np.log10(min_lambda))
plt.xlabel(r"Model polynomial degree")
plt.ylabel('$\log_{10}(\lambda)$'+' of minimum MSE')
plt.title(r"$\lambda$ resulting in smallest MSE ( $\sigma =0.5$, #datapoints = " + str(datapoints) +r' )')
plt.show()

print('Random seed: '+'sum([ord(c) for c in "corona"]' + ',k = {0}, nr_lambdas = {1}, smallest_lambda = {2}, polyMax = {3}, datapoints = {4}'.format(k , nr_lambdas , smallest_lambda , polyMax , datapoints ))

print('Lambda resulting in smallest MSE')
print(min_lambda)


"""
RUN EXAMPLES:

###############################################################################################################################################
Random seed: sum([ord(c) for c in "corona"],k = 7, nr_lambdas = 100, smallest_lambda = -3, polyMax = 7, datapoints = 1000
Lambda resulting in smallest MSE
[0.06681849 0.05723037 0.04658963 0.04663141 0.04258908 0.04270373
 0.04239197]
###############################################################################################################################################

"""