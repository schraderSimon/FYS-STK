import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread
from sklearn import datasets
from function_library import *
"""Set up data"""
np.random.seed(sum([ord(c) for c in "CORONA"]))
terrain = imread("../data/Korea.tif")
datapoints=2000
degree=5

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
eta=0.001
epochs=100
n_hidden_neurons=[10]
n_categories=1
batch_size=5
Lambda=1e-10
nn=NeuralNetwork(X_train_scaled,z_train_scaled,n_hidden_neurons=n_hidden_neurons,n_categories=1,epochs=epochs,batch_size=batch_size,eta=eta,lmbd=Lambda)
nn.train()
prediction_train=nn.predict_probabilities(X_train_scaled)
prediction_test=nn.predict_probabilities(X_test_scaled)
print("Test MSE: %f" %MSE(prediction_test,z_test_scaled))
print("Train MSE: %f" %MSE(prediction_train,z_train_scaled))
#print(prediction_train[:5])
#print(z_train_scaled[:5])
