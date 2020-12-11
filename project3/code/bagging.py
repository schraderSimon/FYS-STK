from sklearn.ensemble import BaggingRegressor
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
n_estimators=[int(1e3)]
test_err_MSE=np.zeros(len(n_estimators),dtype="float")
train_err_MSE=np.zeros(len(n_estimators),dtype="float")

try:
    os.mkdir(mapname)
except:
    pass
plt.title("Decision Tree Bagging Regressor")
plt.xlabel("Number of Bootstraps")
plt.ylabel("MAE (kcal/mol)")
plt.xscale("log")
for index in range(number_crossvals):
    print("Forest %d"%index)
    X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,index)
    T_train=T_train.ravel()
    T_test=T_test.ravel()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    for i,number_estimators in enumerate(n_estimators):
        print("Baum %d"%i)
        regressor=BaggingRegressor(DecisionTreeRegressor(max_depth=20),n_estimators=number_estimators)
        #regressor=DecisionTreeRegressor(max_depth=5)
        regressor.fit(X_train_scaled,T_train)
        train_pred=regressor.predict(X_train_scaled)
        test_pred=regressor.predict(X_test_scaled)
        print(test_pred)
        test_err_MSE[i]+=MAE(test_pred,T_test)

        train_err_MSE[i]+=MAE(T_train,train_pred)
train_err_MSE/=number_crossvals;
test_err_MSE/=number_crossvals;
plt.plot(n_estimators,test_err_MSE, label="test error")
plt.plot(n_estimators,train_err_MSE, label="train error")

plt.legend()
#plt.savefig("../figures/bagging_2.pdf")
print("MSE: Minimum error at  %d estimators with test error %f"%(n_estimators[np.argmin(test_err_MSE)],np.min(test_err_MSE)));

#plt.show()
