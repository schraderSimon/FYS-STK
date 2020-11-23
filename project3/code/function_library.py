import scipy.io as sio
import numpy as np
import sys
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
def sort_coulomb_matrix(A):
    Anew=np.zeros_like(A)
    sorted=np.flip(np.argsort(np.linalg.norm(A,axis=1))) #order of column norms
    for i in range(len(sorted)):
        for j in range(len(sorted)):
            Anew[i][j]=A[sorted[i],sorted[j]]
    return Anew
def remove_hydrogen(coulomb_matrix):
    for i in range(len(coulomb_matrix)):
        if abs(coulomb_matrix[i,i]-0.5)<1e-12:
            coulomb_matrix[i,:]=0
            coulomb_matrix[:,i]=0
    return coulomb_matrix
def find_max_non_hydrogens(X):
    maxcounter=0
    mincounter=23
    for i,coulomb_matrix in enumerate(X):
        counter=0
        for j in range(23):
            if abs(coulomb_matrix[j,j]-0.5)<1e-12:
                counter+=1
        if(counter==0):
            print("I'm just hydrogen.")
            continue
        if(maxcounter<counter):
            maxcounter=counter
        if mincounter>counter:
            mincounter=counter

    print(mincounter,maxcounter)
def convert_dataset(X,R,Z,T,P,index=0,rem_H=False): #Convert dataset to a matrix set that can be used for nonconvol. networks
    """
    takes the input data and converts it to train and test sets based on P
    and the chosen index
    """
    testing_indeces=P[index]
    training_indeces=np.delete(P,index,0).ravel()

    tridiagonal_indices=np.tril_indices(len(X[0][0]))
    Xnew=np.zeros((len(X),int(len(X[0][0])*(len(X[0][0])+1)/2))) # 7165 * 23*24/2 (only lower triangular elements)
    for i,coloumb_matrix in enumerate(X):
        if rem_H==True:
            coloumb_matrix=remove_hydrogen(coloumb_matrix)
        coulomb_ordered=sort_coulomb_matrix(coloumb_matrix)
        Xnew[i]=coulomb_ordered[tridiagonal_indices] # Only take the tridiagonal indeces
    X=Xnew.reshape(len(Xnew),-1)

    T=T.T
    R=R.reshape(len(X),-1)
    Z=Z.reshape(len(X),-1)
    X_train, X_test= createTestTrain(X,training_indeces,testing_indeces)
    R_train, R_test= createTestTrain(R,training_indeces,testing_indeces)
    Z_train, Z_test= createTestTrain(Z,training_indeces,testing_indeces)
    T_train, T_test= createTestTrain(T,training_indeces,testing_indeces)
    return X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test
filedata="../data/qm7.mat"
X,R,Z,T,P=read_data(filedata)
X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,0)
