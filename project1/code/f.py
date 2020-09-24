import numpy as np
from imageio import imread
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *

terrain = imread("../data/Korea.tif")
imageio.imwrite("korea.png",terrain)
plt.figure()
plt.title("Terrain over Parts of Korea")
plt.imshow(terrain, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
datapoints=50000
xs=np.random.randint(len(terrain),size=datapoints)
ys=np.random.randint(len(terrain[1]),size=datapoints)
xy_array=np.column_stack((xs,ys))
z=[]
for x,y in xy_array:
    z.append(terrain[x,y])
z=np.array(z)
print(np.shape(terrain))
maxdeg=20
n_bootstraps=1000
MSE_train_OLS=np.zeros(maxdeg)
MSE_test_OLS=np.zeros(maxdeg)
bias_OLS=np.zeros(maxdeg)
variance_OLS=np.zeros(maxdeg)
R2_train_OLS=np.zeros(maxdeg)
R2_test_OLS=np.zeros(maxdeg)
for deg in range(1,maxdeg+1):
    X=DesignMatrix_deg2(xs,ys,deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.25)
    mean_val=np.mean(z_train)
    z_train_scaled=z_train-mean_val
    z_test_scaled=z_test-mean_val
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    #X_train_scaled=X_train
    #X_test_scaled=X_test
    beta, beta_variance = LinearRegression(X_train_scaled,z_train_scaled)
    z_train_scaled_fit=X_train_scaled@beta
    MSE_train[deg-1]+=(MSE(z_train_scaled,z_train_scaled_fit))
    R2_train[deg-1]+=(R2(z_train_scaled,z_train_scaled_fit))
    z_test_scaled_fit=np.zeros((len(z_test),n_bootstraps))
    for i in range(n_bootstraps):
        X_b, z_b=resample(X_train_scaled,z_train_scaled)
        beta, beta_variance = LinearRegression(X_b,z_b)
        z_test_scaled_fit[:,i]=X_test_scaled @ beta
    MSE_test[deg-1] =bootstrap_MSE(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    bias[deg-1] = bootstrap_bias(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    variance[deg-1] = bootstrap_variance(z_test_scaled,z_test_scaled_fit,n_bootstraps)
    x=np.linspace(0,len(terrain[0]),len(terrain[0]),dtype=int)
    y=np.linspace(0,len(terrain),len(terrain),dtype=int)
    plt.imshow(fit_terrain(x,y,beta,scaler,mean_val,deg),cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("../figures/Norway_fit_deg%d_points%d.png"%(deg,datapoints))
    plt.clf()
"""
x=np.linspace(0,len(terrain),len(terrain),dtype=int)
y=np.linspace(0,len(terrain),len(terrain),dtype=int)
terrain_fit=np.zeros((len(y),len(x)))
leny=len(y)
lenx=len(x)
print(np.shape(terrain_fit))
x, y = np.meshgrid(x,y)
x=x.ravel()
y=y.ravel()
print(scaler.mean_, scaler.var_)
X=scaler.transform(DesignMatrix_deg2(x,y,deg))
z=X @ beta+mean_val
print("z: ")
print(np.shape(z))
for i in range(lenx):
    for j in range(leny):
        terrain_fit[j,i]=z[i*leny+j]
plt.imshow(terrain_fit, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("../figures/Korea_fit_deg%d.png"%deg)
plt.show()
"""
