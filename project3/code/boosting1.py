from function_library import *
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
np.random.seed(272) #L dies after 272 days. RIP L


number_crossvals=5

filedata="../data/qm7.mat"

X,R,Z,T,P=read_data(filedata)
tree_depths=[2,3,4,5,6,7,8]
Ms=[100,200,400,800,1600,3200]
test_err_MSE=np.zeros((len(Ms),len(tree_depths)),dtype="float")
train_err_MSE=np.zeros((len(Ms),len(tree_depths)),dtype="float")
input_type="reduced"
for index in range(number_crossvals):
    print("Crossval %d"%index)
    X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,index)
    T_train=T_train.ravel()
    T_test=T_test.ravel()
    if input_type=="reduced": #Reduced Coulomb matrix
        pass
    if input_type=="noH": #No Hydrogen
        testing_indeces=P[index]
        training_indeces=np.delete(P,index,0).ravel()
        X_removed=create_hydrogenfree_coulomb_matrix(X)
        X_removed=reduce_coulomb(X_removed)
        X_train, X_test= createTestTrain(X_removed,training_indeces,testing_indeces)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    for i,tree_depth in enumerate(tree_depths):
        for j, estimator in enumerate(Ms):
            regressor=xgb.XGBRegressor(objective ='reg:squarederror',
                                       booster="gbtree",max_depth=tree_depth, learning_rate = 50/estimator,
                                       alpha = 10, n_estimators = estimator,n_jobs=4,random_state=272,
                                       colsample_bytree = 0.3, subsample=0.5)
            regressor.fit(X_train_scaled,T_train)
            train_pred=regressor.predict(X_train_scaled)
            test_pred=regressor.predict(X_test_scaled)
            print("Tree depth: %d, M: %d, Error: %f "%(tree_depth,estimator,MAE(test_pred,T_test)))
            test_err_MSE[j,i]+=MAE(test_pred,T_test)

            train_err_MSE[j,i]+=MAE(T_train,train_pred)
train_err_MSE/=number_crossvals;
test_err_MSE/=number_crossvals;
header="2,3,4,5,6,7,8"
np.savetxt("../csvdata/boosting_1test%s.csv"%input_type,test_err_MSE,header=header,delimiter=",")
np.savetxt("../csvdata/boosting_1train%s.csv"%input_type,train_err_MSE,header=header,delimiter=",")
