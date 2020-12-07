import numpy as np
from function_library import *
from sklearn import linear_model
import sys
from sklearn.metrics import mean_absolute_error as MAE
from function_library import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
filedata="../data/qm7.mat"
filename="X_ridge"
outfile="../csvdata/%s.csv"%filename
X,R,Z,T,P=read_data(filedata)
np.set_printoptions(linewidth=200)
if filename=="X_ridge":
    X_removed=reduce_coulomb(X)
elif filename=="Xremoved_ridge":
    X_removed=create_hydrogenfree_coulomb_matrix(X)
    X_removed=reduce_coulomb(X_removed)

PCA_values=np.array(list(range(1,len(X_removed[0])+1)))
train_errors=np.zeros(len(PCA_values))
test_errors=np.zeros_like(train_errors)
T=T.T
Lambdas=np.logspace(-8,1,10)
PCA_matrices=create_PCA_matrices(X_removed,T,P)
for n,number in enumerate(PCA_values):
    for l,Lambda in enumerate(Lambdas):
        test_error,train_error=PCA_ridge_MAE(PCA_matrices,P,Lambda,n)
        if l==0:
            train_errors[n]=train_error
            test_errors[n]=test_error
        else:
            if test_errors[n] < test_error:
                train_errors[n]=train_error
                test_errors[n]=test_error
    print("Number of components: %d, test error: %.2f, train error: %.2f"%(number,test_errors[n],train_errors[n]))
dictionary={"PCA_values":PCA_values,"test_errors":test_errors,"train_errors":train_errors}
df=pd.DataFrame(dictionary)
df.to_csv(outfile)
"""
Use the
"""
