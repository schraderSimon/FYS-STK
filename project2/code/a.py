import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread

from function_library import *
import time
import pandas as pd
import sys
"""
Here, we compare the analytical result for a 5-dimensional problem (Franke Function)
with the results achieved using RMSprop, standard SGD and reduction SGD.
"""
try:
    datapoints=int(sys.argv[1])
    degree=int(sys.argv[2])
    batchsize=int(sys.argv[3])
    num_etas=int(sys.argv[4])
    epochs=int(sys.argv[5])

except:
    datapoints=20000
    degree=18
    batchsize=16
    num_etas=10#array length for etas & t1_values
    epochs=1000
"""Set up data"""
np.random.seed(sum([ord(c) for c in "CORONA"]))
terrain = imread("../data/Korea.tif")
x=np.random.randint(len(terrain),size=datapoints) #random integers
y=np.random.randint(len(terrain[1]),size=datapoints) #random integers for y
xy_array=np.column_stack((x,y))
z=[]
for xv,yv in xy_array:
    z.append(terrain[xv,yv])
z=np.array(z) #data to be fitted
X=DesignMatrix_deg2(x,y,degree,False)
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25) #split in train and test data Design Matrix
z_train_scaled=z_train-np.mean(z_train) #remove mean
z_test_scaled=z_test-np.mean(z_train) #remove mean
scaler=StandardScaler()
scaler.fit(X_train) #Scale
X_train_scaled=scaler.transform(X_train) #scale train Design matrix
X_test_scaled=scaler.transform(X_test) #scale test Design matrix

"""Calculate OLS MSE"""
thetaOLS,unimportant=LinearRegression(X_train_scaled,z_train_scaled)
z_fit=X_test_scaled @ thetaOLS
MSE_OLS=MSE(z_test_scaled,z_fit)

#Part 1: SGDs as a function of starting learning rate (using far too many epochs)
eta_simple=np.logspace(-7,2,num_etas)
MSE_simple=np.zeros_like(eta_simple)
eta_RMSprop=eta_simple
MSE_RMSprop=np.zeros_like(eta_RMSprop)
t1_values=np.logspace(1,10,num_etas)
MSE_decay=np.zeros_like(t1_values)
sgd=SGD(X_train_scaled,z_train_scaled,n_epochs=epochs,batchsize=batchsize)

print("Started SGD")
for i in range(len(eta_simple)):
    theta=sgd.simple_fit(eta=eta_simple[i])
    MSE_simple[i]=MSE(z_test_scaled,X_test_scaled@theta)

print("Started RMSprop")
for i in range(len(eta_RMSprop)):
    theta=sgd.RMSprop(eta=eta_RMSprop[i])
    MSE_RMSprop[i]=MSE(z_test_scaled,X_test_scaled@theta)

print("Started decay_fit")
t0=1000
for i in range(len(t1_values)):
    theta=sgd.decay_fit(t0=t0,t1=t1_values[i])
    MSE_decay[i]=MSE(z_test_scaled,X_test_scaled@theta)
dictionary={"batchsize":batchsize, "datapoints":datapoints,"degree":degree,"analytical_error:":MSE_OLS,"epochs":epochs,"eta":eta_simple,"t0":t0,"t1":t1_values}
dictionary["t1"]=t1_values
dictionary["MSE_SGD"]=MSE_simple
dictionary["MSE_RMSprop"]=MSE_RMSprop
dictionary["MSE_decay"]=MSE_decay
df = pd.DataFrame(dictionary)
df.to_csv("../csvData/OLSMSE_eta_datap%ddeg%dbatchs%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))
plt.plot([t1_values[0],t1_values[-1]],[MSE_OLS,MSE_OLS],label="exact")
plt.plot(t1_values,MSE_decay,label="t0 = %d"%t0)
plt.ylim(0,1e5)
plt.xscale("log")
plt.legend()
plt.savefig("1a.png")

plt.clf()

plt.plot(eta_simple,MSE_simple,label="SGD")
plt.plot(eta_RMSprop,MSE_RMSprop,label="RMSprop")
plt.plot([eta_simple[0],eta_simple[-1]],[MSE_OLS,MSE_OLS],label="exact")
plt.ylim(0,1e5)
plt.xscale("log")
plt.legend()
plt.savefig("1b.png")

plt.clf()
ideal_eta_SGD=eta_simple[np.nanargmin(MSE_simple)-2]
ideal_eta_RMSprop=min(eta_RMSprop[np.nanargmin(MSE_RMSprop)],0.1)
ideal_t1=t1_values[np.nanargmin(MSE_decay)] #DONT USE ARGMIN IT FUCKS WITH YOU HERE!!!!!

#Part 2: SGDs as a function of numbers of epochs
Part2=True
if(Part2):
    start=0; end=4; amount=5;
    amounts=np.logspace(start,end,amount)
    MSE_amount_simple=np.zeros_like(amounts)
    MSE_amount_RMSprop=np.zeros_like(amounts)
    MSE_amount_decay=np.zeros_like(amounts)
    sgd=SGD(X_train_scaled,z_train_scaled,batchsize=batchsize)
    for i in range(len(amounts)):
        sgd.n_epochs=int(amounts[i]) #My programming teacher would have killed me for this
        MSE_amount_simple[i]=MSE(z_test_scaled,X_test_scaled@sgd.simple_fit(eta=ideal_eta_SGD))
        MSE_amount_RMSprop[i]=MSE(z_test_scaled,X_test_scaled@sgd.RMSprop(eta=ideal_eta_RMSprop))
        MSE_amount_decay[i]=MSE(z_test_scaled,X_test_scaled@sgd.decay_fit(t1=ideal_t1,t0=t0))
    dictionary={"batchsize":batchsize, "datapoints":datapoints,"degree":degree,"analytical_error:":MSE_OLS,"eta_RMSprop":ideal_eta_RMSprop,"eta_SGD":ideal_eta_SGD,"t0":t0,"t1":ideal_t1}
    dictionary["epochs"]=amounts
    dictionary["MSE_SGD"]=MSE_amount_simple
    dictionary["MSE_RMSprop"]=MSE_amount_RMSprop
    dictionary["MSE_decay"]=MSE_amount_decay
    df = pd.DataFrame(dictionary)

    df.to_csv("../csvData/OLSMSE_epochs_datap%ddeg%dbatchs%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))

    plt.plot(amounts,MSE_amount_RMSprop,label="RMSProp")
    plt.plot(amounts,MSE_amount_decay,label="decay")
    plt.plot(amounts,MSE_amount_simple,label="SGD")
    plt.plot([amounts[0],amounts[-1]],[MSE_OLS,MSE_OLS],label="exact")
    plt.ylim(0,1e5)
    plt.xscale("log")
    plt.legend()
    plt.savefig("2.png")
    plt.clf()

#Part 3: SGDs as a function of batch size
Part3=True
if(Part3):
    batchsizes=np.array([2**i for i in range(8)]) # 1 to 128
    MSE_batches_simple=np.zeros(len(batchsizes))
    MSE_batches_RMSprop=np.zeros(len(batchsizes))
    MSE_batches_decay=np.zeros(len(batchsizes))
    sgd=SGD(X_train_scaled,z_train_scaled,n_epochs=epochs)
    time_batches=np.zeros(len(batchsizes))
    for i in range(len(batchsizes)):
        start=time.time_ns()
        sgd.batchsize=int(batchsizes[i]) #My programming teacher would have killed me for this
        MSE_batches_simple[i]=MSE(z_test_scaled,X_test_scaled@sgd.simple_fit(eta=ideal_eta_SGD))
        MSE_batches_RMSprop[i]=MSE(z_test_scaled,X_test_scaled@sgd.RMSprop(eta=ideal_eta_RMSprop))
        MSE_batches_decay[i]=MSE(z_test_scaled,X_test_scaled@sgd.decay_fit(t1=ideal_t1,t0=t0))
        end=time.time_ns()
        time_batches[i]=float((end-start))*1e-9
    dictionary={"batchsize":batchsize, "datapoints":datapoints,"epochs":epochs,"degree":degree,"analytical_error:":MSE_OLS,"eta_RMSprop":ideal_eta_RMSprop,"eta_SGD":ideal_eta_SGD,"t0":t0,"t1":ideal_t1}
    dictionary["time"]=time_batches
    dictionary["MSE_SGD"]=MSE_batches_simple
    dictionary["MSE_RMSprop"]=MSE_batches_RMSprop
    dictionary["MSE_decay"]=MSE_batches_decay
    df = pd.DataFrame(dictionary)

    df.to_csv("../csvData/OLSMSE_batches_datap%ddeg%dbatches%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))

    plt.plot(batchsizes,MSE_batches_RMSprop,label=r"RMSProp, $\eta=%.2e$"%ideal_eta_RMSprop)
    plt.plot(batchsizes,MSE_batches_decay,label="decay, t0=%d, t1=%d"%(t0,int(ideal_t1)))
    plt.plot(batchsizes,MSE_batches_simple,label="SGD, $\eta=%.2e$"%ideal_eta_SGD)
    plt.plot([batchsizes[0],batchsizes[-1]],[MSE_OLS,MSE_OLS],label="exact")
    plt.ylim(MSE_OLS*0.9,0.6*1e5)
    plt.xscale("log")
    plt.legend()
    plt.savefig("3.png")
    plt.clf()

#Part 4: Use these parameters for Ridge regression
"""Find the ideal ridge lambda paramter"""
Part4=True
if(Part4):

    lambda_val = np.logspace(-30,-15,4)
    MSE_test_kfoldRidge_lambda = np.zeros(len(lambda_val))
    for i in range(len(lambda_val)):
        MSE_test_kfoldRidge_lambda[i] = KCrossValRidgeMSE(X,z,5,lambda_val[i])
    optimal_lambda=lambda_val[np.argmin(MSE_test_kfoldRidge_lambda)]
    theta_exact_Ridge,unimportant=RidgeRegression(X_train_scaled,z_train_scaled,optimal_lambda,False) #the beta values
    z_fit=X_test_scaled @ theta_exact_Ridge
    MSE_Ridge=MSE(z_test_scaled,z_fit)
    MSE_Ridge_lambda_eta_simple=np.zeros((num_etas,len(lambda_val)))
    MSE_Ridge_lambda_eta_RMSprop=np.zeros((num_etas,len(lambda_val)))
    eta_simple=np.logspace(-7,2,num_etas)
    sgd=SGD_Ridge(X_train_scaled,z_train_scaled,n_epochs=epochs,batchsize=32,Lambda=1)
    for j in range(len(lambda_val)):
        sgd.Lambda=0
        for i in range(num_etas):
            theta_simple=sgd.simple_fit(eta=eta_simple[i])
            theta_RMSprop=sgd.RMSprop(eta=eta_simple[i])
            MSE_Ridge_lambda_eta_simple[i,j]=MSE(z_test_scaled,X_test_scaled@theta_simple)
            MSE_Ridge_lambda_eta_RMSprop[i,j]=MSE(z_test_scaled,X_test_scaled@theta_RMSprop)
        dictionary={"batchsize":batchsize, "datapoints":datapoints,"analytical_error:":MSE_Ridge,"degree":degree,"epochs":epochs,"eta":eta_simple,"t0":t0,"t1":t1_values}
        dictionary["MSE_SGD"]=MSE_Ridge_lambda_eta_simple[:,j]
        dictionary["MSE_RMSprop"]=MSE_Ridge_lambda_eta_RMSprop[:,j]
        df = pd.DataFrame(dictionary)
        df.to_csv("../csvData/RidgeMSE_lambda%.3e_eta_datap%ddeg%dbatchs%detas%depochs%d.csv"%(lambda_val[j],datapoints,degree,batchsize,num_etas,epochs))
        plt.plot(eta_simple,MSE_Ridge_lambda_eta_simple[:,j],label="SGD,lambda=%.3e"%lambda_val[j])
        plt.plot(eta_RMSprop,MSE_Ridge_lambda_eta_RMSprop[:,j],label="RMSprop")
    plt.plot([eta_simple[0],eta_simple[-1]],[MSE_Ridge,MSE_Ridge],label="exact")
    plt.ylim(0,1e5)
    plt.xscale("log")
    plt.legend()
    plt.savefig("4.png")
