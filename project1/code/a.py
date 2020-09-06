from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from help_functions import *
n=3 #Data points for x and y
maxdeg=5
x=np.linspace(0,1,n).reshape(-1, 1)
y=np.linspace(0,1,n).reshape(-1, 1)
x,y= np.meshgrid(x,y)
z=FrankeFunction(x,y)
print(z)
print(x)
p=round((maxdeg+1)*(maxdeg+2)/2)
X=np.zeros((n*n,p))
X[:,0]=1
X[:,1]=np.ravel(x) # Adds x on the first column
X[:,2]=np.ravel(y) # Adds x on the first column
count=3
for i in range(2,maxdeg+1):
    for j in range(i+1):
        X[:,count]=X[:,2]**j*X[:,1]**(i-j)
        print(f"count:{count} x: {j} y: {i-j}")
        count+=1;
for i in range(p):
    print(i)
    print(X[:,i])
