from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

minNrk = 2
maxNrk = 10
k = np.array([x for x in range(minNrk,maxNrk+1)])

mindeg = 2
maxdeg = 8

datapoints=400
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
z=FrankeFunction(x,y)+np.random.normal(0,0.3, datapoints)

MSE_test_kfold = np.zeros((len(k),maxdeg-mindeg+1))
j = 0
for deg in range(mindeg,maxdeg+1):
    X = DesignMatrix_deg2(x,y,deg)
    for i in range(len(k)):
       MSE_test_kfold[i,j] = (KCrossValMSE(X,z,k[i])+ KCrossValMSE(X,z,k[i]) +KCrossValMSE(X,z,k[i])+KCrossValMSE(X,z,k[i])+KCrossValMSE(X,z,k[i]))/5
    j +=1

j = 0
for s in range(maxNrk-minNrk+1):
    plt.plot(list(range(mindeg,maxdeg+1)),MSE_test_kfold[s,:],label = 'k =' +str(k[j]))
    j+=1
plt.xlabel('Polynomial degree')
plt.ylabel('MSE')
plt.title('Mean Square Error of OLS using different numbers (k) of splits (#datapoints =' +str(datapoints)+r',$\sigma = 0.3$ )')
plt.legend()
plt.show()

"""
if (csv_polydegree_comp):
    #OUTPUTS CSV FILE CONTAINING MSE COMPARISONS BETWEEN BOOTSTRAP, KFOLD-OLS AND KFOLD-RIDGE OVER A SPAN OF POLYNOMIAL DEGREES (SAMPLE TYPE 1)
    dict = {'polynomial degree': list(range(mindeg,maxdeg +1)),'MSE Bootstrap': MSE_test_boot, 'MSE Kfold-crossvalidation': MSE_test_kfold,'MSE Kfold-crossvalidation Ridge': MSE_test_kfoldRidge}
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/KfoldRidge_polydegree_comparison_error.csv')

if (csv_Lambdaval_comp):
    #OUTPUTS CSV FILE CONTAINING MSE OF KFOLD-RIDGE OVER A SPAN OF LAMBDA VALUES (SAMPLE TYPE 2)
    dict = {'Lambda value': lambda_val,'MSE Kfold-crossvalidation Ridge': MSE_test_kfoldRidge_lambda}
    df = pd.DataFrame(dict)
    df.to_csv('../csvData/RidgeRegression_Lambda_error.csv')
    plt.plot(np.log10(lambda_val),MSE_test_kfoldRidge_lambda)
    plt.show()

"""
