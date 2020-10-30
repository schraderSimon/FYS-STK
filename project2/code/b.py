import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread
from sklearn import datasets
from function_library import *
import pandas as pd
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
"""Set up data"""
np.random.seed(sum([ord(c) for c in "CORONA"]))
terrain = imread("../data/Korea.tif")
datapoints=2000
degree=10

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
z_test_scaled.shape=[len(z_test_scaled),1]
z_train_scaled.shape=[len(z_train_scaled),1]

scaler=StandardScaler()
scaler.fit(X_train) #Scale
X_train_scaled=scaler.transform(X_train) #scale train Design matrix
X_test_scaled=scaler.transform(X_test) #scale test Design matrix

"""Set up parameters"""
runscikit=False
eta=0.01
epochs=100
n_hidden_neurons=[100,100,50,50]
n_hidden_layers=len(n_hidden_neurons)
n_categories=1
batch_size=200
amount_lambdas=10
amount_etas=8
Lambdas=np.logspace(-16,3,amount_lambdas)
etas=np.logspace(-6,1,amount_etas)
Lambda_eta_error_train=np.zeros((amount_lambdas,amount_etas))
Lambda_eta_error_test=np.zeros((amount_lambdas,amount_etas))
Scikit_Lambda_eta_error_train=np.zeros((amount_lambdas,amount_etas))
Scikit_Lambda_eta_error_test=np.zeros((amount_lambdas,amount_etas))

activation_function_type="LeakyRELU"
solver="ADAM"
nn=NeuralNetwork(X_train_scaled,z_train_scaled,
    n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
    n_categories=n_categories,epochs=epochs,batch_size=batch_size,
    activation_function_type=activation_function_type,
    errortype="MSE",solver=solver) #Create Neural Network

for l, Lambda in enumerate(Lambdas):
    for e, eta in enumerate(etas):
        print("Eta: %.2e"%eta)
        print("Lambda: %.2e"%Lambda)
        testErr, trainErr, testR2, trainR2= Crossval_Neural_Network(5, nn, eta, Lambda,X,z) #Perform Cross Validation with the Neural Network code
        print("Test MSE Neural Network: %f" %testErr)
        print("Train MSE Neutral Network: %f" %trainErr)
        Lambda_eta_error_test[l,e]=testErr
        Lambda_eta_error_train[l,e]=trainErr
        if activation_function_type=="sigmoid":
            activation_function_type="logistic" #it is called differently in scikit learn.
        if runscikit:
            testErr, trainErr, testR2, trainR2= CrossVal_Regression(5,eta,Lambda,X,z,activation_function_type.lower(),solver.lower(),n_hidden_neurons,epochs) #Perform Cross Validation with the Neural Network code
            Scikit_Lambda_eta_error_test[l,e]=testErr
            Scikit_Lambda_eta_error_train[l,e]=trainErr
            print("Test MSE Scikit Learn: %f"%testErr)
            print("Train MSE Scikit Learn: %f"%trainErr)


"""write to file"""
import os
mapname="../csvData/%db1_%s%s%d%d_epoch"%(len(n_hidden_neurons),activation_function_type,solver,datapoints,degree)
try:
    os.mkdir(mapname)
except:
    pass
outfile=open("%s/info.txt"%mapname,"w")
outfile.write("lambda,")
for Lambda in Lambdas:
    outfile.write("%e,"%Lambda)
outfile.write("\neta,")
for eta in etas:
    outfile.write("%e,"%eta)
outfile.write("\nhidden_neuron,")
for neuron in n_hidden_neurons:
    outfile.write("%d,"%neuron)
outfile.write("\ndatapoints,%d,\ndegree,%d,\nepoch, %d,\nbatch_size,%d\nactivation_function, %s,\nsolver,%s,\n"%(datapoints,degree,epochs,batch_size,activation_function_type,solver))
pd.DataFrame(Scikit_Lambda_eta_error_train).to_csv("%s/Scikit_train.csv"%mapname)
pd.DataFrame(Scikit_Lambda_eta_error_test).to_csv("%s/Scikit_test.csv"%mapname)
pd.DataFrame(Lambda_eta_error_train).to_csv("%s/train.csv"%mapname)
pd.DataFrame(Lambda_eta_error_test).to_csv("%s/test.csv"%mapname)
min_test_error=np.inf
min_train_error=np.inf

for Lambda in np.logspace(-30,-10,20):
    test_error_inversion,train_error_inversion=KCrossValRidgeMSE(X,z,5,Lambda)
    if test_error_inversion<min_test_error:
        min_test_error=test_error_inversion
        min_train_error=train_error_inversion
outfile.write("train_error_inversion,%d,\n"%min_train_error)
outfile.write("test_error_inversion,%d,\n"%min_test_error)
outfile.close()
