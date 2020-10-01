from imageio import imread
import imageio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler,MinMaxScaler
from small_function_library import *
import numpy as np
import pandas as pd
import sys
terrain = imread("../data/Korea.tif") #terrain data
np.random.seed(sum([ord(c) for c in "CORONA"]))

try:
    filename=sys.argv[1]
except:
    filename="Korea20000.csv" #File containing the relevant data
try:
    scaling_factor=step=7 #how much the resulting images should be scaled. This cannot be one, otherwise the machine crashes (too large matrix)
except:
    scaling_factor=step=7
data=pd.read_csv("../csvData/%s"%filename)
mindeg=data["mindeg"][0];maxdeg=data["maxdeg"][0]
k=data["k"][0]
datapoints=data["datapoints"][0]
n_bootstraps=data["n_bootstrap"][0]
n_bootstraps_lasso=data["n_bootstraps_lasso"][0]
x=np.random.randint(len(terrain),size=datapoints) #random points for x
y=np.random.randint(len(terrain[1]),size=datapoints) #random points for y
xy_array=np.column_stack((x,y))
z=[]
for xv,yv in xy_array:
    z.append(terrain[xv,yv])
z=np.array(z)
ideal_degree_OLS=np.argmin(data["MSEkfoldOLS"])
ideal_degree_RIDGE=np.argmin(data["MSEkfoldRIDGE"])
ideal_lambda_RIDGE=data["ideal_lambda_RIDGE"][ideal_degree_RIDGE]
ideal_degree_OLS+=data["mindeg"][0]; #S
ideal_degree_RIDGE+=data["mindeg"][0];
X_OLS=DesignMatrix_deg2(x,y,ideal_degree_OLS)
z_mean=np.mean(z)
z_scaled=z-z_mean
scaler=StandardScaler()
scaler.fit(X_OLS)
X_OLS_scaled=scaler.transform(X_OLS)
betaOLS, beta_varianceOLS = LinearRegression(X_OLS_scaled,z_scaled)


px=np.arange(0,len(terrain),step,dtype=int)
py=np.arange(0,len(terrain),step,dtype=int)
terrain_fit_OLS=np.zeros((len(py),len(px)))
terrain_fit_Ridge=np.zeros((len(py),len(px)))
terrain_scaled=np.zeros((len(py),len(px)))
lenpx=len(px)
lenpy=len(py)
px, py = np.meshgrid(px,py)
px=px.ravel()
py=py.ravel()
X=scaler.transform(DesignMatrix_deg2(px,py,ideal_degree_OLS))
z=X @ betaOLS+z_mean
for i in range(lenpx):
    for j in range(lenpy):
        terrain_fit_OLS[j,i]=z[i*lenpy+j]
        terrain_scaled[j,i]=terrain[j*step,i*step]
fig, (ax0, ax1,ax2) = plt.subplots(ncols=3,figsize=(9, 3))
mapstyle="rainbow"
ax2.set_title("Original")
ax0.set_title("OLS degree %d"%(ideal_degree_OLS))


scaler=StandardScaler()
X_RIDGE=DesignMatrix_deg2(x,y,ideal_degree_RIDGE)
scaler.fit(X_RIDGE)
X_RIDGE_scaled=scaler.transform(X_RIDGE)
betaRidge, beta_varianceRidge=RidgeRegression(X_RIDGE_scaled,z_scaled,ideal_lambda_RIDGE)
X=scaler.transform(DesignMatrix_deg2(px,py,ideal_degree_RIDGE))
z=X @ betaRidge+z_mean
for i in range(lenpx):
    for j in range(lenpy):
        terrain_fit_Ridge[j,i]=z[i*lenpy+j]
ax1.set_title(r"Ridge degree %d, $\lambda$=%.3e"%(ideal_degree_RIDGE,ideal_lambda_RIDGE))
_min, _max = -100, np.max(terrain_scaled)
im1=ax1.imshow(terrain_fit_Ridge, cmap=mapstyle,vmin = _min, vmax = _max)
im0=ax0.imshow(terrain_fit_OLS, cmap=mapstyle,vmin = _min, vmax = _max)
im2=ax2.imshow(terrain_scaled, cmap=mapstyle,vmin = _min, vmax = _max)
cbar=plt.colorbar(im2,ax=ax2)
cbar.set_label("Height [m]")
plt.savefig("../figures/Fitted_images%s.pdf"%filename[:-4])
plt.show()

"""
run as python3 create_picture.py filename scaling_factor
for example python3 create_picture.py Korea50000_NOBOOTSTRAP.csv 7


"""
