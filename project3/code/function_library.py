import scipy.io as sio
import numpy as np
def read_data(filename):
    data=sio.loadmat(filename)
    X_data=data["X"]; #this is the 7165*23*23 array
    R_data=data["R"]; #These are the positions: 7165*23*3
    Z_data=data["Z"]; #These are the charges: 7165*23
    T_data=data["T"]; #This is the atomization energy!! 1*7165
    P_data=data["P"]; # This is just a split for 5-fold cross validation 5*1433
    return X_data, R_data, Z_data, T_data, P_data
def createTestTrain(matrix,training_indeces,testing_indeces):
    return matrix[training_indeces], matrix[testing_indeces]

def convert_dataset(X,R,Z,T,P,index=0): #Convert dataset to a matrix set that can be used for nonconvol. networks
    """
    takes the input data and converts it to train and test sets based on P
    and the chosen index
    """
    testing_indeces=P[index]
    training_indeces=np.delete(P,index,0).ravel()
    X=X.reshape(len(X),-1)
    T=T.T
    R=R.reshape(len(X),-1)
    Z=Z.reshape(len(X),-1)
    X_train, X_test= createTestTrain(X,training_indeces,testing_indeces)
    R_train, R_test= createTestTrain(R,training_indeces,testing_indeces)
    Z_train, Z_test= createTestTrain(Z,training_indeces,testing_indeces)
    T_train, T_test= createTestTrain(T,training_indeces,testing_indeces)
    return X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test
