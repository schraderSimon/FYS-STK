from function_library import *
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
np.random.seed(272) #L dies after 272 days. RIP L


number_crossvals=5 # The number of cross validations to perform. 5 to get "actual" results.

filedata="../data/qm7.mat"
input_type="noH" #noH or reduced
X,R,Z,T,P=read_data(filedata)
tree_depth=20
n=np.linspace(1,tree_depth,tree_depth,dtype="int")
test_err_MAE=np.zeros(len(n),dtype="float") #Minimized via MAE
train_err_MAE=np.zeros(len(n),dtype="float")
test_err_MSE=np.zeros(len(n),dtype="float") #Minimized via MSE
train_err_MSE=np.zeros(len(n),dtype="float")

plt.title("Decision Tree Regressor")
plt.xlabel("Maximum Tree depth")
plt.ylabel("MAE (kcal/mol)")
plt.xticks(np.arange(1,tree_depth+1))

for index in range(number_crossvals):
    X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,index)
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
    for depth in range(tree_depth): #For each tree depth
        """Create MSE-reducing tree"""
        regressor=DecisionTreeRegressor(max_depth=depth+1,criterion='mse')
        regressor.fit(X_train_scaled,T_train)
        train_pred=regressor.predict(X_train_scaled)
        test_pred=regressor.predict(X_test_scaled)
        test_err_MSE[depth]+=MAE(test_pred,T_test)
        train_err_MSE[depth]+=MAE(T_train,train_pred)
        """Create MAE-reducing tree"""
        regressor=DecisionTreeRegressor(max_depth=depth+1,criterion='mae')
        regressor.fit(X_train_scaled,T_train)
        train_pred=regressor.predict(X_train_scaled)
        test_pred=regressor.predict(X_test_scaled)
        test_err_MAE[depth]+=MAE(test_pred,T_test)
        train_err_MAE[depth]+=MAE(T_train,train_pred)
train_err_MAE/=number_crossvals;
test_err_MAE/=number_crossvals;
train_err_MSE/=number_crossvals;
test_err_MSE/=number_crossvals;

plt.plot(n,test_err_MSE, label="test error, trained with MSE")
plt.plot(n,train_err_MSE, label="train error, trained with MSE")
plt.plot(n,test_err_MAE, label="test error, trained with MAE")
plt.plot(n,train_err_MAE, label="train error, trained with MAE")

plt.legend()
plt.savefig("../figures/decision_tree%s.pdf"%input_type)
print("MAE: Minimum error at depth %d with test error %f"%(np.argmin(test_err_MAE),np.min(test_err_MAE)));
print("MSE: Minimum error at depth %d with test error %f"%(np.argmin(test_err_MSE),np.min(test_err_MSE)));

plt.show()
"""
run as
python3 decision_tree.py
"""
