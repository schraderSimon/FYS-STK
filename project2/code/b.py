import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread
from sklearn import datasets
from function_library import *
from sklearn.linear_model import LinearRegression as LinReg
"""Set up data"""
np.random.seed(sum([ord(c) for c in "CORONA"]))
terrain = imread("../data/Korea.tif")
datapoints=2000
degree=7

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
eta=0.01
epochs=3000
n_hidden_neurons=[100,100]
n_hidden_layers=len(n_hidden_neurons)
n_categories=1
batch_size=100
Lambda=0
etas=np.logspace(-7,3,23)
activation_function_type="sigmoid"
for eta in etas:
    print("Eta: %.2e"%eta)
    nn=NeuralNetwork(X_train_scaled,z_train_scaled,
        n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
        n_categories=n_categories,epochs=epochs,batch_size=batch_size,eta=eta,lmbd=Lambda,
        activation_function_type=activation_function_type,
        errortype="MSE",solver="RMSProp")
    nn.train()
    prediction_train=nn.predict_probabilities(X_train_scaled)
    prediction_test=nn.predict_probabilities(X_test_scaled)
    print("Test MSE Neural Network: %f" %MSE(prediction_test,z_test_scaled))
    print("Train MSE Neutral Network: %f" %MSE(prediction_train,z_train_scaled))
#print(prediction_train[:5])
#print(z_train_scaled[:5])
thetaOLS,unimportant=LinearRegression(X_train_scaled,z_train_scaled)
print("Test MSE inversion: %f "%MSE(z_test_scaled,X_test_scaled @ thetaOLS))
print("Train MSE inversion: %f "%MSE(z_train_scaled,X_train_scaled @ thetaOLS))
reg = LinReg(fit_intercept=False).fit(X_train_scaled, z_train_scaled)

print("Test MSE inversion Scikit: %f "%MSE(z_test_scaled,reg.predict(X_test_scaled)))
print("Train MSE inversion Scikit: %f "%MSE(z_train_scaled,reg.predict(X_train_scaled)))
