from function_library import *
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
np.random.seed(272) #L dies after 272 days. RIP L


number_crossvals=5

filedata="../data/qm7.mat"

X,R,Z,T,P=read_data(filedata)
Ms=np.array(list(range(20,6401,20)),dtype=int)
alphas=[1,5,10]
etas=[0.05,0.01]
test_err_MSE=np.zeros((len(Ms),len(alphas)*len(etas)),dtype="float")
train_err_MSE=np.zeros((len(Ms),len(alphas)*len(etas)),dtype="float")
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
    dtrain=xgb.DMatrix(X_train_scaled,label=T_train)
    dtest=xgb.DMatrix(X_test_scaled,label=T_test)
    for i,alpha in enumerate(alphas):
        for k, eta in enumerate(etas):
            print("%d/%d"%(len(etas)*i+k,len(etas)*len(alphas)))
            param = {'max_depth': 5, 'eta': eta, "objective" :'reg:squarederror'}
            param["random_state"]=272
            param["booster"]="gbtree"
            param["colsample_bytree"]=0.3
            param["subsample"]=0.5
            param["alpha"]=alpha
            bst=xgb.train(param,dtrain,0)# create the
            for j in range(len(Ms)):
                bst=xgb.train(param,dtrain,20,xgb_model=bst)
                train_pred=bst.predict(dtrain)
                test_pred=bst.predict(dtest)
                print(j/len(Ms))
                test_err_MSE[j,len(etas)*i+k]+=MAE(test_pred,T_test)
                train_err_MSE[j,len(etas)*i+k]+=MAE(T_train,train_pred)
train_err_MSE/=number_crossvals;
test_err_MSE/=number_crossvals;
header=""
for alpha in alphas:
    for eta in etas:
        header+="%dalpha %.2feta,"%(alpha,eta)
np.savetxt("../csvdata/boosting_2test%s.csv"%input_type,test_err_MSE,header=header,delimiter=",")
np.savetxt("../csvdata/boosting_2train%s.csv"%input_type,train_err_MSE,header=header,delimiter=",")
