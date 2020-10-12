from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import time

""" Parameters Begin"""
np.random.seed(sum([ord(c) for c in "corona"]))
minNrk = 10
maxNrk = 10
n_bootstraps=20000
maxdeg=6
datapoints=1000
""" Parameters end"""

k = np.array([i for i in range(minNrk,maxNrk+1)])
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)
z=FrankeFunction(x,y)+0.1*np.random.normal(0,1, datapoints)

MSE_test_kfold = np.zeros((maxdeg,len(k)))
MSE_test_boot = np.zeros(maxdeg)

timeElapsed_bootstrap = np.zeros(maxdeg)
timeElapsed_kfold = np.zeros(maxdeg)

idx = 0
idx_2 = 0
for j in k:
    for deg in range(1,maxdeg+1):
        X=DesignMatrix_deg2(x,y,deg,True)
        #start = time.time()#----- Used for timing the methods
        MSE_test_kfold[deg-1,idx] = KCrossValMSE(X,z,j)
        #end = time.time()#----- Used for timing the methods
        #timeElapsed_kfold[idx_2] = end-start#----- Used for timing the methods
        if (idx > 0):
            continue
        X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
        z_train_scaled=z_train#-np.mean(z_train)
        z_test_scaled=z_test#-np.mean(z_train)
        #scaler=StandardScaler()
        #scaler.fit(X_train)
        X_train_scaled=X_train
        X_test_scaled=X_test
        #X_train_scaled=scaler.transform(X_train)
        #X_test_scaled=scaler.transform(X_test)
        beta, beta_variance = LinearRegression(X_train_scaled,z_train_scaled)
        z_train_scaled_fit=X_train_scaled@beta
        z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
        #start = time.time()   #----- Used for timing the methods
        for i in range(n_bootstraps):
            X_b, z_b=resample(X_train_scaled,z_train_scaled)
            beta, beta_variance = LinearRegression(X_b,z_b)
            z_test_scaled_fit[:,i]=X_test_scaled @ beta
        MSE_test_boot[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
        #end = time.time()#----- Used for timing the methods
        #timeElapsed_bootstrap[idx_2] = end-start#----- Used for timing the methods
        idx_2 +=1
    idx +=1
#print('time bootstrap')
#print(timeElapsed_bootstrap)
#print('time kfoldcross')
#print(timeElapsed_kfold)
plt.plot(list(range(1,maxdeg+1)),MSE_test_boot,label = 'Bootstrap')
for i in range(len(k)):
    plt.plot(list(range(1,maxdeg+1)),MSE_test_kfold[:,i],label = str(k[i])+' fold CV')
#for i in range(maxdeg):#----- Used for timing the methods
#    plt.annotate('%.3f s' % timeElapsed_bootstrap[i],(list(range(1,maxdeg+1))[i],MSE_test_boot[i]),color ='blue')#----- Used for timing the methods
#    plt.annotate('%.3f s' % timeElapsed_kfold[i],(list(range(1,maxdeg+1))[i],MSE_test_kfold[i]),color ='orange')#----- Used for timing the methods
plt.legend()
plt.xlabel('polynomial degree')
plt.ylabel('MSE')
plt.title('Comparing Bootstrap to k- fold CV (#Bootstraps = '+str(n_bootstraps)+', #datapoints = '+str(datapoints)+')')
#plt.title('Comparing Bootstrap to 10- fold CV elapsed time in seconds and MSE (#Bootstraps = '+str(n_bootstraps)+', #datapoints = '+str(datapoints)+')')
plt.show()

#dict = {'polynomial degree': list(range(1,maxdeg +1)),'MSE Bootstrap': MSE_test_boot, 'MSE Kfold-crossvalidation': MSE_test_kfold}

#df = pd.DataFrame(dict)

#df.to_csv('C:/Users/adria/Documents/Studier/FYS-STK4155/FYS-STK/project1/csvData/Polydeg_Kfold_error.csv')

