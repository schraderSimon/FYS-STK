from sklearn.ensemble import RandomForestRegressor
from function_library import *
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
np.random.seed(272) #L dies after 272 days. RIP L


number_crossvals=5

filedata="../data/qm7.mat"

X,R,Z,T,P=read_data(filedata)
tree_depth=9
n_estimators=np.logspace(0,3.5,8,dtype=int)
Types=["25full","25sqrt","25log","25third","9full","9sqrt","9log","9third","5full","5sqrt","5log","5third"]
header="25full,25sqrt,25log,25third,9full,9sqrt,9log,9third,5full,5sqrt,5log,5third"
sizes=[25,15,5]
maxfeatures=["auto","sqrt","log2",1/3]
test_errors=np.zeros((len(n_estimators),len(Types)),dtype="float")
train_errors=np.zeros((len(n_estimators),len(Types)),dtype="float")

for index in range(number_crossvals):
    X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,index)
    T_train=T_train.ravel()
    T_test=T_test.ravel()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    for i,number_estimators in enumerate(n_estimators):
        for j, size in enumerate(sizes):
            for k, maxfeature in enumerate(maxfeatures):
                regressor=RandomForestRegressor(max_depth=size,max_features=maxfeature,n_estimators=number_estimators,n_jobs=4)
                regressor.fit(X_train_scaled,T_train)
                train_pred=regressor.predict(X_train_scaled)
                test_pred=regressor.predict(X_test_scaled)
                test_errors[i,j*4+k]+=MAE(test_pred,T_test)
                train_errors[i,j*4+k]+=MAE(T_train,train_pred)
test_errors/=number_crossvals;
train_errors/=number_crossvals;
np.savetxt("../csvdata/randomforest_%etest.csv"%n_estimators[-1],test_errors,header=header,delimiter=",")
np.savetxt("../csvdata/randomforest_%etrain.csv"%n_estimators[-1],train_errors,header=header,delimiter=",")
