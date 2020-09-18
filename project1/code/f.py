import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *

terrain = imread("../data/Korea.tif")
plt.figure()
plt.title("Terrain over Parts of Korea")
plt.imshow(terrain, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
datapoints=3000
xs=np.random.randint(len(terrain),size=datapoints)
ys=np.random.randint(len(terrain[1]),size=datapoints)
xy_array=np.column_stack((xs,ys))
z=[]
for x,y in xy_array:
    z.append(terrain[x,y])
z=np.array(z)
print(np.shape(terrain))
maxdeg=7
n_bootstraps=100
MSE_train=np.zeros(maxdeg)
MSE_test=np.zeros(maxdeg)
bias=np.zeros(maxdeg)
variance=np.zeros(maxdeg)
R2_train=np.zeros(maxdeg)
R2_test=np.zeros(maxdeg)
for deg in range(maxdeg,maxdeg+1):
    X=DesignMatrix_deg2(xs,ys,deg)
    #print(X)
    #X_shuffled=X.copy()
    #np.random.shuffle(X_shuffled)
    #print(X_shuffled)
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
    print(f"Degree: {deg}")
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
from numba import jit
@jit
def fit_func(beta,x,y,polydegree,mean,inverse_var,include_intercept=False):
    adder=0 #The matrix dimension is increased by one if include_intercept is True
    p=round((polydegree+1)*(polydegree+2)/2)-1 #The total amount of coefficients
    if include_intercept:
        p+=1
        adder=1
    func=np.zeros(p)
    if include_intercept:
        func[0]=1
    func[0+adder]=(x-mean[0+adder])*inverse_var[0+adder] # Adds x on the first column
    func[1+adder]=(y-mean[1+adder])*inverse_var[1+adder] # Adds y on the second column
    count=2+adder
    xpot=[x**j for j in range(polydegree+1)]
    ypot=[y**j for j in range(polydegree+1)]
    for i in range(2,polydegree+1):
        for j in range(i+1):
            func[count]=(xpot[j]*ypot[i-j]-mean[count])*inverse_var[count]
            count+=1;
    z=func @ beta
    return z
def fit_terrain(x,y,beta,scaler,mean_valz,degree=5):
    mean=scaler.mean_
    var=scaler.scale_
    terrain_fit=np.zeros((len(y),len(x)))
    leny=len(y)
    lenx=len(x)
    inverse_var=1/var
    for i in range(lenx):
        print(i)
        for j in range(leny):
            terrain_fit[i][j]=fit_func(beta,x[i],y[j],degree,mean,inverse_var)+mean_valz
    return terrain_fit
x=np.linspace(0,len(terrain),len(terrain),dtype=int)
y=np.linspace(0,len(terrain),len(terrain),dtype=int)
plt.imshow(fit_terrain(x,y,beta,scaler,mean_val,maxdeg),cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("../figures/Korea_fit_deg%d.png"%deg)
plt.show()
