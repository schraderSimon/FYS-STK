from function_library import *
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
np.random.seed(272) #L dies after 272 days. RIP L


number_crossvals=1

filedata="../data/qm7.mat"

X,R,Z,T,P=read_data(filedata)
tree_depth=13
n=np.linspace(1,tree_depth,tree_depth,dtype="int")
test_err_MAE=np.zeros(len(n),dtype="float")
train_err_MAE=np.zeros(len(n),dtype="float")

try:
    os.mkdir(mapname)
except:
    pass

for index in range(number_crossvals):
    counter=0
    X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,index)
    meanie=np.mean(T_train)
    T_train=-T_train; T_train=T_train-meanie
    T_test=-T_test; T_test=T_test-meanie
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    for depth in range(tree_depth):
        print("Baum")
        regressor=DecisionTreeRegressor(max_depth=depth+1,criterion='mse')
        regressor.fit(X_train_scaled,T_train)
        train_pred=regressor.predict(X_train_scaled)
        test_pred=regressor.predict(X_test_scaled)
        test_err_MAE[depth]+=MAE(test_pred,T_test)
        train_err_MAE[depth]+=MAE(T_train,train_pred)
    train_err_MAE/=number_crossvals;
    test_err_MAE/=number_crossvals;
    plt.plot(n,test_err_MAE, label="test")
    plt.plot(n,train_err_MAE, label="train")
    plt.title("Decision Tree Regressor")
    plt.xlabel("Maximum Tree depth")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()
