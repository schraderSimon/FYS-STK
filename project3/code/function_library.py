import scipy.io as sio
import numpy as np
import sys
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def read_data(filename):
    """Reads the relevant data from a .mat-file"""
    data=sio.loadmat(filename)
    X_data=data["X"]; #this is the 7165*23*23 array
    R_data=data["R"]; #These are the positions: 7165*23*3
    Z_data=data["Z"]; #These are the charges: 7165*23
    T_data=data["T"]; #This is the atomization energy!! 1*7165
    P_data=data["P"]; # This is just a split for 5-fold cross validation 5*1433
    return X_data, R_data, Z_data, T_data, P_data
def createTestTrain(matrix,training_indeces,testing_indeces):
    """For a given matrix and a list of training and testing indeces, returns train and test matrix"""
    return matrix[training_indeces], matrix[testing_indeces]
def sort_coulomb_matrix(A):
    """Sorts a coulom matrix by their column norm in descending order"""
    Anew=np.zeros_like(A)
    sorted=np.flip(np.argsort(np.linalg.norm(A,axis=1))) #order of column norms
    for i in range(len(sorted)):
        for j in range(len(sorted)):
            Anew[i][j]=A[sorted[i],sorted[j]]
    return Anew

def RidgeRegression(X_training,y_training,Lambda, include_beta_variance=True):

    """
    Performs Ridge Regression. Implemented as SVD

    Parameters:
    X_training (2D array), The design matrix X,
    y_training (array), the targets Y
    Lambda (double), regularization parameter,
    include_beta_variance (bool), whether the variance should be included too.

    Returns:
    beta (array), the Ridge regression beta
    beta_variance (2D array),The R.R. var (0 if include_beta_variance is False)

    """
    I = np.eye(len(X_training[0,:]))
    if include_beta_variance:
        inverse_matrix = np.linalg.inv(X_training.T @ X_training+Lambda*I)
        beta_variance = np.diagonal(inverse_matrix)
    else:
        beta_variance=0
    u, s, vh = np.linalg.svd(X_training, full_matrices=False)
    smat=np.zeros((vh.shape[0],u.shape[1]))
    for i in range(len(s)):
        smat[i][i]=s[i]
    beta= vh.T @ (np.linalg.inv(smat.T@smat+(I*Lambda)) @ smat.T) @ u.T @ y_training
    return beta, beta_variance

def create_PCA_matrices(X,Y,P,k=5):
    """
    Creates a set of k train and test matrices that are scaled and have PCA performed.

    Input:
    X (2D matrix): The design matrix
    Y: (1D matrix): The targets
    P: The Cross-Validation split
    k: The number of Cross-validation splits

    Returns:
    X_trains: A list with k Training matrices, scaled and full PCA performed on them
    X_tests: A list with k Testing matrices, scaled and full PCA performed on them
    Y_trains: A list with k target lists for training
    Y_trains: A list with k target lists for testing
    """
    scaler = StandardScaler()
    X_trains=[]
    X_tests=[]
    Y_trains=[]
    Y_tests=[]
    num_PC=len(X[0])
    for i in range(k):
        testing_indeces=P[i]
        training_indeces=np.delete(P,i,0).ravel()
        X_training, X_testing= createTestTrain(X,training_indeces,testing_indeces)
        Y_train, Y_test= createTestTrain(Y,training_indeces,testing_indeces)
        Y_training=Y_train-np.mean(Y_train)
        Y_testing=Y_test-np.mean(Y_test)
        #Scale X
        scaler.fit(X_training)
        X_training_scaled = scaler.transform(X_training)
        X_testing_scaled = scaler.transform(X_testing)
        #perform Ridge regression
        pca = PCA(n_components=num_PC)
        pca.fit(X_training_scaled)
        X_train_PCA=pca.transform(X_training_scaled)
        X_test_PCA=pca.transform(X_testing_scaled)
        X_trains.append(X_train_PCA)
        X_tests.append(X_test_PCA)
        Y_trains.append(Y_training)
        Y_tests.append(Y_testing)
    return X_trains, X_tests, Y_trains, Y_tests


def PCA_ridge_MAE(PCA_matrices,P,Lambda,num_PC=10,k=5):
    MAE_crossval = np.zeros(k)
    MAE_crossval_train=np.zeros(k)

    for i in range(k):
        X_train_PCA=PCA_matrices[0][i][:,:num_PC]
        X_test_PCA=PCA_matrices[1][i][:,:num_PC]
        Y_train_PCA=PCA_matrices[2][i]
        Y_test_PCA=PCA_matrices[3][i]
        beta, beta_variance = RidgeRegression(X_train_PCA,Y_train_PCA,Lambda,False)
        #print(beta)
        Y_training_fit = X_train_PCA @ beta
        Y_testing_fit = X_test_PCA @ beta
        #calculate MSE for each fold
        MAE_crossval[i] = MAE(Y_test_PCA,Y_testing_fit)
        MAE_crossval_train[i] = MAE(Y_train_PCA,Y_training_fit)
    MAE_estimate = np.mean(MAE_crossval)
    MAE_train_estimate=np.mean(MAE_crossval_train)

    return MAE_estimate, MAE_train_estimate
def create_hydrogenfree_coulomb_matrix(X):
    """Takes the coulomb matrix and creates hydrogen-free coulomb matrix.

    The function takes the R-matrix and the Z-matrix and
    creates a Coulomb matrix where all hydrogen atoms, as well as the hydrogen
    molecule, are removed.
    Input:
    X (2D matrix, n_atoms*n_atoms) - Coulomb matrix WITH hydrogen in it
    Returns:
    Coulomb matrix (2D, n_nonhydrongen_atoms*n_nonhydrongen_atoms)
    """
    max_hydrogen,min_hydrogen=find_max_non_hydrogens(X)
    X_new=np.zeros((len(X),max_hydrogen,max_hydrogen),dtype=float)
    for coul_number,coulomb_matrix in enumerate(X):
        remove_indexes=[] #The indexes of Hydrogen
        for i in range(23):
            if abs(coulomb_matrix[i,i]-0.5)<1e-12: #If it's hydrogen
                remove_indexes.append(i)
            elif abs(coulomb_matrix[i,i])<1e-12: #if it's empty
                remove_indexes.append(i)
        c_del=np.delete(coulomb_matrix, remove_indexes, 0)
        c_del=np.delete(c_del, remove_indexes, 1)
        adder=max_hydrogen-len(c_del)
        c_del=np.pad(c_del, ((0,adder),(0,adder)), mode='constant', constant_values=0)
        X_new[coul_number]=c_del
    return X_new
def find_max_non_hydrogens(X):
    """Finds the maximum and the minimum number of non-hydrogen"""
    maxcounter=0
    mincounter=23
    for i,coulomb_matrix in enumerate(X):
        counter=0 #the number of hydrogens and empties
        for j in range(23):
            if abs(coulomb_matrix[j,j]-0.5)<1e-12: #if it's hydrogen
                counter+=1
            elif abs(coulomb_matrix[j,j])<1e-12: #if it's empty
                counter+=1
        non_hydrogens=23-counter
        if(maxcounter<non_hydrogens):
            maxcounter=non_hydrogens
        if mincounter>non_hydrogens:
            mincounter=non_hydrogens
    return maxcounter, mincounter
def reduce_coulomb(X):
    """Takes a set of Coulomb Matrices X and returns the ordered upper triangular 2D version"""
    tridiagonal_indices=np.tril_indices(len(X[0][0]))
    Xnew=np.zeros((len(X),int(len(X[0][0])*(len(X[0][0])+1)/2))) # 7165 * 23*24/2 (only lower triangular elements)
    for i,coloumb_matrix in enumerate(X):
        coulomb_ordered=sort_coulomb_matrix(coloumb_matrix)
        Xnew[i]=coulomb_ordered[tridiagonal_indices] # Only take the tridiagonal indeces
    return Xnew.reshape(len(Xnew),-1)
def convert_dataset(X,R,Z,T,P,index=0): #Convert dataset to a matrix set that can be used for nonconvol. networks
    """
    takes the input data and converts it to train and test sets based on P
    and the chosen index
    """
    testing_indeces=P[index]
    training_indeces=np.delete(P,index,0).ravel()
    X=reduce_coulomb(X)
    T=T.T
    R=R.reshape(len(X),-1)
    Z=Z.reshape(len(X),-1)
    X_train, X_test= createTestTrain(X,training_indeces,testing_indeces)
    R_train, R_test= createTestTrain(R,training_indeces,testing_indeces)
    Z_train, Z_test= createTestTrain(Z,training_indeces,testing_indeces)
    T_train, T_test= createTestTrain(T,training_indeces,testing_indeces)
    return X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test
