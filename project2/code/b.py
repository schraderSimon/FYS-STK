import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread
from sklearn import datasets
from function_library import *
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
eta=0.01
epochs=500
n_hidden_neurons=[100,50]
n_hidden_layers=len(n_hidden_neurons)
n_categories=1
batch_size=50
amount_lambdas=2
amount_etas=2
Lambdas=np.logspace(-16,0,amount_lambdas)
etas=np.logspace(-3,-2,amount_etas)
Lambda_eta_error_train=np.zeros((amount_lambdas,amount_etas))
Lambda_eta_error_test=np.zeros((amount_lambdas,amount_etas))

activation_function_type="RELU"
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
        testErr, trainErr, testR2, trainR2= CrossVal_Regression(5,eta,Lambda,X,z,activation_function_type.lower(),solver.lower(),n_hidden_neurons) #Perform Cross Validation with the Neural Network code
        print("Test MSE Scikit Learn: %f"%testErr)
        print("Train MSE Scikit Learn: %f"%trainErr)
"""
for l, Lambda in enumerate(Lambdas):
    for e, eta in enumerate(etas):
        print("Eta: %.2e"%eta)
        print("Lambda: %.2e"%Lambda)
        nn.update_parameters_reset(eta=eta,lmbd=Lambda)
        nn.train()
        prediction_train=nn.predict_probabilities(X_train_scaled)
        prediction_test=nn.predict_probabilities(X_test_scaled)
        Lambda_eta_error_test[l,e]=MSE_test=MSE(prediction_test,z_test_scaled)
        Lambda_eta_error_train[l,e]=MSE_train=MSE(prediction_train,z_train_scaled)
        print("Test MSE Neural Network: %f" %MSE_test)
        print("Train MSE Neutral Network: %f" %MSE_train)
"""


thetaOLS,unimportant=LinearRegression(X_train_scaled,z_train_scaled) #Compare with linear regression
print("Test MSE inversion: %f "%MSE(z_test_scaled,X_test_scaled @ thetaOLS))
print("Train MSE inversion: %f "%MSE(z_train_scaled,X_train_scaled @ thetaOLS))
reg = LinReg(fit_intercept=False).fit(X_train_scaled, z_train_scaled)

print("Test MSE inversion Scikit: %f "%MSE(z_test_scaled,reg.predict(X_test_scaled)))
print("Train MSE inversion Scikit: %f "%MSE(z_train_scaled,reg.predict(X_train_scaled)))
#Note: Using [100,100], eta=[10^-3 to 10^-1], lambda=0, 1000 epochs, batchsize 100, 20.000 datapoints
#degree 18 and no regularization, I got the error below 10,000 (sigmoid, RMSProp)
print("Minimal Test error:")
print(np.nanmin(Lambda_eta_error_test))
print("Minimal Train error")
print(np.nanmin(Lambda_eta_error_train))
