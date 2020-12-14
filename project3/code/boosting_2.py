from function_library import *
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
np.random.seed(272) #L dies after 272 days. RIP L


number_crossvals=1 # The number of cross validations to perform. 5 to get "actual" results.


filedata="../data/qm7.mat"
printfile=True #If data shall be written to file
X,R,Z,T,P=read_data(filedata)
evals=100 #how often the MAE should be written to file

#Ms=np.array(list(range(20,6401,20)),dtype=int)
Ms=np.array(list(range(evals,1000,evals)),dtype=int)
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
    dtrain=xgb.DMatrix(X_train_scaled,label=T_train) #XGB compatible format
    dtest=xgb.DMatrix(X_test_scaled,label=T_test) #XGB compatible format
    for i,alpha in enumerate(alphas):
        for k, eta in enumerate(etas):
            param = {'max_depth': 5, 'eta': eta, "objective" :'reg:squarederror'}
            param["random_state"]=272
            param["tree_method"]="gpu_hist"
            param["sampling_method"]="gradient_based"
            param["booster"]="gbtree"
            param["colsample_bytree"]=0.3
            param["subsample"]=0.5
            param["alpha"]=alpha
            bst=xgb.train(param,dtrain,0)# create the model (untrained)
            for j in range(len(Ms)):
                bst=xgb.train(param,dtrain,evals,xgb_model=bst) #train model with evals steps
                train_pred=bst.predict(dtrain) #predict train error
                test_pred=bst.predict(dtest) #predict test error
                test_err_MSE[j,len(etas)*i+k]+=MAE(test_pred,T_test)
                train_err_MSE[j,len(etas)*i+k]+=MAE(T_train,train_pred)
train_err_MSE/=number_crossvals;
test_err_MSE/=number_crossvals;
header=""
for alpha in alphas:
    for eta in etas:
        header+="%dalpha %.2feta,"%(alpha,eta)
if(len(alphas)*len(etas)==1):
    test_err_MSE=test_err_MSE.ravel()
    print("Minimum value at %d with error %f"%(Ms[np.argmin(test_err_MSE)],np.min(test_err_MSE)))
if(printfile):
    np.savetxt("../csvdata/boosting_2test%s_ex.csv"%input_type,test_err_MSE,header=header,delimiter=",")
    np.savetxt("../csvdata/boosting_2train%s_ex.csv"%input_type,train_err_MSE,header=header,delimiter=",")
"""
Params for  boosting_2testreduced.csv:
    20 - 6401, step length 20,
    param = {'max_depth': 5, 'eta': eta, "objective" :'reg:squarederror'}
    param["random_state"]=272
    param["booster"]="gbtree"
    param["colsample_bytree"]=0.3
    param["subsample"]=0.5
    param["alpha"]=alpha
    alphas=[1,5,10]
    etas=[0.05,0.01]
"""
"""
run as
python3 boosting_2.py
"""
