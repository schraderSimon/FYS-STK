import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread

from function_library import *
import time
import pandas as pd
import sys
try:
    datapoints=int(sys.argv[1])
    degree=int(sys.argv[2])
    batchsize=int(sys.argv[3])
    num_etas=int(sys.argv[4])
    epochs=int(sys.argv[5])
    max_learning_rate=float(sys.argv[6])
except:
    datapoints=2000
    degree=10
    batchsize=16
    num_etas=10#array length for etas & t1_values
    epochs=1000
    max_learning_rate=0.1

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

#Calculate OLS MSE
thetaOLS,unimportant=LinearRegression(X_train_scaled,z_train_scaled) #We do not look at standard deviation here.
z_fit=X_test_scaled @ thetaOLS
MSE_OLS=MSE(z_test_scaled,z_fit) #The analytical solution.

#Part 1: SGDs as a function of starting learning rate (using far too many epochs)
eta_simple=np.logspace(-7,2,num_etas) #eta from 1e-7 to 1e2
MSE_simple=np.zeros_like(eta_simple)
eta_RMSprop=eta_simple #same as eta_simple
eta_adam=eta_simple #same as eta_simple
MSE_RMSprop=np.zeros_like(eta_RMSprop)
MSE_adam=np.zeros_like(eta_adam)
t1_values=np.logspace(1,10,num_etas) #t1 from 1e1 to 1e10
MSE_decay=np.zeros_like(t1_values)
sgd=SGD(X_train_scaled,z_train_scaled,n_epochs=epochs,batchsize=batchsize) #create SGD object

for i in range(len(eta_simple)):
    """For each learning rate in the eat list, create a simple fit"""
    theta=sgd.simple_fit(eta=eta_simple[i]); sgd.reset()
    MSE_simple[i]=MSE(z_test_scaled,X_test_scaled@theta)

for i in range(len(eta_RMSprop)):
    """For each learning rate in the eta list, create a RMSprop fit"""
    theta=sgd.RMSprop(eta=eta_RMSprop[i]) ; sgd.reset()
    MSE_RMSprop[i]=MSE(z_test_scaled,X_test_scaled@theta)

for i in range(len(eta_adam)):
    """For each learning rate in the eta list, create an ADAM fit"""
    theta=sgd.ADAM(eta=eta_adam[i]) ; sgd.reset()
    MSE_adam[i]=MSE(z_test_scaled,X_test_scaled@theta)

t0=1000 #initiate t0 as 1000
for i in range(len(t1_values)):
    """For each learning rate in the t1 list, create a fit"""
    theta=sgd.decay_fit(t0=t0,t1=t1_values[i]) ; sgd.reset()
    MSE_decay[i]=MSE(z_test_scaled,X_test_scaled@theta)

#create a dictionary to write to file
dictionary={"batchsize":batchsize, "datapoints":datapoints,"degree":degree,"analytical_error":MSE_OLS,"epochs":epochs,"eta":eta_simple,"t0":t0,"t1":t1_values}
dictionary["t1"]=t1_values
dictionary["MSE_SGD"]=MSE_simple
dictionary["MSE_RMSprop"]=MSE_RMSprop
dictionary["MSE_decay"]=MSE_decay
dictionary["MSE_ADAM"]=MSE_adam

df = pd.DataFrame(dictionary)
df.to_csv("../csvData/OLSMSE_eta_datap%ddeg%dbatchs%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))

#The "ideal" values for the learning rates. Will be reused here
ideal_eta_SGD=min(eta_simple[np.nanargmin(MSE_simple)-1],max_learning_rate)
ideal_eta_RMSprop=min(eta_RMSprop[np.nanargmin(MSE_RMSprop)-1],max_learning_rate)
ideal_eta_adam=min(eta_adam[np.nanargmin(MSE_adam)-1],max_learning_rate)
ideal_t1=t1_values[np.nanargmin(MSE_decay)]


#Part 2: SGDs as a function of numbers of epochs
Part2=True
if(Part2):
    start=0; end=4; amount=end-start+1;
    amounts=np.logspace(start,end,amount)
    MSE_amount_simple=np.zeros_like(amounts)
    MSE_amount_RMSprop=np.zeros_like(amounts)
    MSE_amount_decay=np.zeros_like(amounts)
    MSE_amount_adam=np.zeros_like(amounts)
    sgd=SGD(X_train_scaled,z_train_scaled,batchsize=batchsize)
    for i in range(len(amounts)):
        """For each amount of epochs, calculate the MSE using all 4 methods"""
        sgd.n_epochs=int(amounts[i]) #My programming teacher would have killed me for this
        sgd.reset()
        MSE_amount_simple[i]=MSE(z_test_scaled,X_test_scaled@sgd.simple_fit(eta=ideal_eta_SGD)); sgd.reset()
        MSE_amount_RMSprop[i]=MSE(z_test_scaled,X_test_scaled@sgd.RMSprop(eta=ideal_eta_RMSprop)); sgd.reset()
        MSE_amount_decay[i]=MSE(z_test_scaled,X_test_scaled@sgd.decay_fit(t1=ideal_t1,t0=t0)); sgd.reset()
        MSE_amount_adam[i]=MSE(z_test_scaled,X_test_scaled@sgd.ADAM(eta=ideal_eta_adam)); sgd.reset()

    #create a dictionary to write to file
    dictionary={"batchsize":batchsize, "datapoints":datapoints,"degree":degree,"analytical_error":MSE_OLS,"eta_ADAM":ideal_eta_adam,"eta_RMSprop":ideal_eta_RMSprop,"eta_SGD":ideal_eta_SGD,"t0":t0,"t1":ideal_t1}
    dictionary["epochs"]=amounts
    dictionary["MSE_SGD"]=MSE_amount_simple
    dictionary["MSE_RMSprop"]=MSE_amount_RMSprop
    dictionary["MSE_decay"]=MSE_amount_decay
    dictionary["MSE_ADAM"]=MSE_amount_adam
    df = pd.DataFrame(dictionary)

    df.to_csv("../csvData/OLSMSE_epochs_datap%ddeg%dbatchs%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))


#Part 3: SGDs as a function of batch size
Part3=True
if(Part3):
    batchsizes=np.array([2**i for i in range(8)]) # 1 to 128
    MSE_batches_simple=np.zeros(len(batchsizes))
    MSE_batches_RMSprop=np.zeros(len(batchsizes))
    MSE_batches_ADAM=np.zeros(len(batchsizes))
    MSE_batches_decay=np.zeros(len(batchsizes))
    time_batches=np.zeros(len(batchsizes))
    sgd=SGD(X_train_scaled,z_train_scaled,n_epochs=epochs,batchsize=int(batchsizes[0]))
    for i in range(len(batchsizes)):
        """For each batch size, calculate the MSE using all 4 methods"""
        start=time.time_ns()
        sgd=SGD(X_train_scaled,z_train_scaled,n_epochs=epochs,batchsize=int(batchsizes[i]))
        MSE_batches_simple[i]=MSE(z_test_scaled,X_test_scaled@sgd.simple_fit(eta=ideal_eta_SGD)); sgd.reset()
        MSE_batches_RMSprop[i]=MSE(z_test_scaled,X_test_scaled@sgd.RMSprop(eta=ideal_eta_RMSprop)); sgd.reset()
        MSE_batches_ADAM[i]=MSE(z_test_scaled,X_test_scaled@sgd.ADAM(eta=ideal_eta_adam)); sgd.reset()
        end=time.time_ns()
        time_batches[i]=float((end-start))*1e-9
    dictionary={"batchsize":batchsizes, "datapoints":datapoints,"epochs":epochs,"degree":degree,"analytical_error":MSE_OLS,"eta_ADAM":ideal_eta_adam,"eta_RMSprop":ideal_eta_RMSprop,"eta_SGD":ideal_eta_SGD,"t0":t0,"t1":ideal_t1}
    dictionary["time"]=time_batches
    dictionary["MSE_SGD"]=MSE_batches_simple
    dictionary["MSE_RMSprop"]=MSE_batches_RMSprop
    dictionary["MSE_ADAM"]=MSE_batches_ADAM
    df = pd.DataFrame(dictionary)
    df.to_csv("../csvData/OLSMSE_batches_datap%ddeg%dbatches%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))


"""
Example how to run:
python3 a.py 2000 10 16 10 1000 0.1
"""
