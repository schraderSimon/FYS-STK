from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(sum([ord(c) for c in "corona"]))
#np.random.seed(670)

k = 4
n_bootstraps=5000

nr_lambdas = 100
min_lambda = -6
max_lambda=-2
lambda_val = np.logspace(min_lambda,0,nr_lambdas)

mindeg = 5
maxdeg = 5

datapoints=200
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
z=FrankeFunction(x,y)+np.random.normal(0,0.1, datapoints)

MSE_test_kfoldLASSO_lambda = np.zeros(nr_lambdas)

MSE_test_kfoldLASSO = np.zeros(maxdeg-mindeg +1)
MSE_test_kfold = np.zeros(maxdeg-mindeg +1)
MSE_test_boot = np.zeros(maxdeg-mindeg +1)
for deg in range(mindeg,maxdeg+1):
    X=DesignMatrix_deg2(x,y,deg)
    MSE_test_kfoldLASSO[deg-mindeg] = KCrossValLASSOMSE(X,z,k,min_lambda)
    for i in range(nr_lambdas):
        MSE_test_kfoldLASSO_lambda[i] = KCrossValLASSOMSE(X,z,k,lambda_val[i])
MSE_test_kfoldLASSO_lambda_smooth = ArraySmoother(MSE_test_kfoldLASSO_lambda,10)
plt.plot(np.log10(lambda_val),(MSE_test_kfoldLASSO_lambda),'r--')
plt.plot(np.log10(lambda_val),(MSE_test_kfoldLASSO_lambda_smooth))
plt.show()
