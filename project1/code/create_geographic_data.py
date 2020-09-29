import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

"""make the data"""
filename="Korea"
terrain = imread("../data/Korea.tif")
np.random.seed(sum([ord(c) for c in "CORONA"]))
datapoints=20000
x=np.random.randint(len(terrain),size=datapoints)
y=np.random.randint(len(terrain[1]),size=datapoints)
xy_array=np.column_stack((x,y))
z=[]
for xv,yv in xy_array:
    z.append(terrain[xv,yv])
z=np.array(z)

"""Set initial parameters"""
n_bootstraps=1000; n_bootstraps_lasso = int(0.05*n_bootstraps)+2
nr_lambdas_ridge = 100; nr_lambdas_lasso=3
min_lambda = -20
max_lambda = -12
mindeg = 1
try:
    maxdeg=int(sys.argv[1])
except:
    maxdeg=5
lasso_tol=0.3 #
lasso_iterations=1e1
k=5
degrees=maxdeg-mindeg+1
"""Create arrays to store data in"""
lambda_val_ridge = np.logspace(min_lambda,max_lambda,nr_lambdas_ridge)
lambda_val_lasso=np.logspace(min_lambda,max_lambda,nr_lambdas_lasso)
MSEkfoldLASSO = np.zeros(degrees);MSEkfoldOLS = np.zeros(degrees);MSEkfoldRIDGE = np.zeros(degrees)
MSEBOOTLASSO = np.zeros(degrees);MSEBOOTRIDGE = np.zeros(degrees);MSEBOOTOLS = np.zeros(degrees)
R2BOOTLASSO = np.zeros(degrees) ;R2BOOTRIDGE = np.zeros(degrees) ;R2BOOTOLS = np.zeros(degrees)
bias_LASSO=np.zeros(degrees);bias_OLS=np.zeros(degrees); bias_RIDGE=np.zeros(degrees)
variance_LASSO=np.zeros(degrees);variance_0LS=np.zeros(degrees) ; variance_Ridge=np.zeros(degrees)
MSE_test_kfoldLASSO_lambda=np.zeros(nr_lambdas_lasso);MSE_test_kfoldRidge_lambda=np.zeros(nr_lambdas_ridge)
ideal_lambda_RIGDE_arr=np.zeros(degrees); ideal_lambda_LASSO_arr=np.zeros(degrees)

"""
def do_bootstrap(n_bootstrap,n_bootstraps_lasso,length_ztest,X,X_test,z,ideal_lambda_LASSO,ideal_lambda_RIGDE,lasso_tol,lasso_iterations):
    z_test_scaled_fit_OLS=np.zeros((length_ztest,n_bootstraps))
    z_test_scaled_fit_LASSO=np.zeros((length_ztest,n_bootstraps_lasso))
    z_test_scaled_fit_RIDGE=np.zeros((length_ztest,n_bootstraps))
    for i in range(n_bootstraps):
        X_b, z_b=resample(X,z)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit_OLS[:,i]=X_test @ beta
        beta, beta_variance=RidgeRegression(X_b,z_b,ideal_lambda_RIGDE)
        z_test_scaled_fit_RIDGE[:,i]=X_test @ beta
        if i >= n_bootstraps_lasso:
            continue
        beta= LASSORegression(X_b,z_b,ideal_lambda_LASSO,lasso_tol,lasso_iterations)
        z_test_scaled_fit_LASSO[:,i]=X_test_scaled @ beta
"""
for deg in range(mindeg,maxdeg+1):
    print("Degree:"+str(deg))
    X=DesignMatrix_deg2(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    z_train_scaled=z_train-np.mean(z_train)
    z_test_scaled=z_test-np.mean(z_train)
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    z_test_scaled_fit_OLS=np.zeros((len(z_test),n_bootstraps))
    z_test_scaled_fit_LASSO=np.zeros((len(z_test),n_bootstraps_lasso))
    z_test_scaled_fit_RIDGE=np.zeros((len(z_test),n_bootstraps))
    for i in range(nr_lambdas_lasso):
        """Find the ideal lambda value using K-fold Cross validation"""
        MSE_test_kfoldLASSO_lambda[i] = KCrossValLASSOMSE(X,z,k,lambda_val_lasso[i],lasso_tol,lasso_iterations)
    for i in range(nr_lambdas_ridge):
        MSE_test_kfoldRidge_lambda[i] = KCrossValRidgeMSE(X,z,k,lambda_val_ridge[i])
    ideal_lambda_LASSO=lambda_val_lasso[np.argmin(MSE_test_kfoldLASSO_lambda)]
    ideal_lambda_RIGDE=lambda_val_ridge[np.argmin(MSE_test_kfoldRidge_lambda)]
    ideal_lambda_RIGDE_arr[deg-mindeg]=ideal_lambda_RIGDE; ideal_lambda_LASSO_arr[deg-mindeg]=ideal_lambda_LASSO

    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit_OLS[:,i]=X_test_scaled @ beta
        beta, beta_variance=RidgeRegression(X_b,z_b,ideal_lambda_RIGDE)
        z_test_scaled_fit_RIDGE[:,i]=X_test_scaled @ beta
        if i >= n_bootstraps_lasso:
            continue
        beta= LASSORegression(X_b,z_b,ideal_lambda_LASSO,lasso_tol,lasso_iterations)
        z_test_scaled_fit_LASSO[:,i]=X_test_scaled @ beta
    """Calculate bootstrap MSE"""
    MSEBOOTOLS[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit_OLS,n_bootstraps)
    MSEBOOTLASSO[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit_LASSO,n_bootstraps_lasso)
    MSEBOOTRIDGE[deg-mindeg] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit_RIDGE,n_bootstraps)
    """Calculate bootstrap R2"""
    R2BOOTLASSO[deg-mindeg] = bootstrap_r2(z_test_scaled,z_test_scaled_fit_LASSO,n_bootstraps_lasso)
    R2BOOTRIDGE[deg-mindeg] = bootstrap_r2(z_test_scaled,z_test_scaled_fit_RIDGE,n_bootstraps)
    R2BOOTOLS[deg-mindeg] = bootstrap_r2(z_test_scaled,z_test_scaled_fit_OLS,n_bootstraps)
    """Calculate bootstrap variance"""
    variance_0LS[deg-mindeg] = bootstrap_variance(z_test_scaled,z_test_scaled_fit_OLS,n_bootstraps)
    variance_Ridge[deg-mindeg] = bootstrap_variance(z_test_scaled,z_test_scaled_fit_RIDGE,n_bootstraps)
    variance_LASSO[deg-mindeg] = bootstrap_variance(z_test_scaled,z_test_scaled_fit_LASSO,n_bootstraps_lasso)
    """Calculate bootstrap bias"""
    bias_OLS[deg-mindeg] = bootstrap_bias(z_test_scaled,z_test_scaled_fit_OLS,n_bootstraps)
    bias_RIDGE[deg-mindeg] = bootstrap_bias(z_test_scaled,z_test_scaled_fit_RIDGE,n_bootstraps)
    bias_LASSO[deg-mindeg] = bootstrap_bias(z_test_scaled,z_test_scaled_fit_LASSO,n_bootstraps_lasso)
    """Calculate k-fold MSE"""
    MSEkfoldRIDGE[deg-mindeg] = KCrossValRidgeMSE(X,z,k,ideal_lambda_RIGDE)
    MSEkfoldLASSO[deg-mindeg] = KCrossValLASSOMSE(X,z,k,ideal_lambda_LASSO,lasso_tol,lasso_iterations)
    MSEkfoldOLS[deg-mindeg] = KCrossValOLSMSE(X,z,k)
write_csv=True
if (write_csv):
    #OUTPUTS CSV FILE CONTAINING MSE OF KFOLD-RIDGE OVER A SPAN OF LAMBDA VALUES (SAMPLE TYPE 2)
    dict = {"nr_lambdas_ridge":nr_lambdas_ridge,'nr_lambdas_lasso':nr_lambdas_lasso,'min_lambda':min_lambda,'max_lambda':max_lambda,"k":k ,'datapoints':datapoints,'n_bootstrap': n_bootstraps,'n_bootstraps_lasso': n_bootstraps_lasso, "mindeg":mindeg, "maxdeg":maxdeg}
    dict["k"]=k; dict["lasso_tol"]=lasso_tol;dict["lasso_iterations"]=lasso_iterations
    dict["R2_lasso"]=R2BOOTLASSO; dict["R2_ridge"]=R2BOOTRIDGE; dict["R2_OLS"]=R2BOOTOLS
    dict["MSEBOOTRIDGE"]=MSEBOOTRIDGE;dict["MSEBOOTLASSO"]=MSEBOOTLASSO;dict["MSEBOOTOLS"]=MSEBOOTOLS
    dict["MSEkfoldRIDGE"]=MSEkfoldRIDGE;dict["MSEkfoldLASSO"]=MSEkfoldLASSO;dict["MSEkfoldOLS"]=MSEkfoldOLS
    dict["bias_LASSO"]=bias_LASSO;dict["bias_RIDGE"]=bias_RIDGE;dict["bias_OLS"]=bias_OLS;
    dict["variance_0LS"]=variance_0LS;dict["variance_Ridge"]=variance_Ridge;dict["variance_LASSO"]=variance_LASSO
    dict["ideal_lambda_LASSO"]=ideal_lambda_LASSO_arr
    dict["ideal_lambda_RIDGE"]=ideal_lambda_RIGDE_arr
    df = pd.DataFrame(dict)
    filepath='../csvData/%s.csv'%filename
    df.to_csv(filepath)
    print("written to %s"%filepath)
