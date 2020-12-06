from sklearn import linear_model
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
from function_library import *
from sklearn.decomposition import PCA

filedata="../data/qm7.mat"
X,R,Z,T,P=read_data(filedata)
find_max_non_hydrogens(X)
X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,0,True)
print(X_train.shape)
T_train=-T_train
T_test=-T_test
T_testr=T_test#-np.mean(T_test)
#sys.exit(1)
X_train_reduced=X_train#[:500,:].copy()
T_train_reduced=T_train#[:500,:].copy()
print(X_train_reduced.shape)
scaler = StandardScaler()
scaler.fit(X_train_reduced)
X_train_reduced_scaled=scaler.transform(X_train_reduced)
X_test_scaled=scaler.transform(X_test)

pca = PCA(n_components=5)
pca.fit(X_train_reduced_scaled)
X_train_reduced_scaled=pca.transform(X_train_reduced_scaled)
X_test_scaled=pca.transform(X_test_scaled)

for alpha in np.logspace(-20,0,40):
    reg=linear_model.Ridge(alpha=alpha,normalize=True)
    reg.fit(X_train_reduced_scaled, T_train_reduced.ravel())
    yfit=reg.predict(X_test_scaled)
    print(MAE(yfit,T_testr.ravel()))
