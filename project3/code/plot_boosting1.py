import numpy as np
import matplotlib.pyplot as plt
from function_library import *
import pandas as pd
import matplotlib
types=["g","c","k--","r","b--","m--","y"]
matplotlib.rcParams.update({"font.size":20})
tree_depths=[2,3,4,5,6,7,8]
plot_trees=tree_depths#[3,5,7,8]
Ms=[100,200,400,800,1600,3200]
test_data=np.loadtxt("../csvdata/randomforest_1testreduced.csv",skiprows=1,dtype="float",delimiter=",")
train_data=np.loadtxt("../csvdata/randomforest_1trainreduced.csv",skiprows=1,dtype="float",delimiter=",")
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of learners M")
plt.title("Test Error")
plt.xscale("log",base=2)
plt.ylim(0,15)
for j in tree_depths:
    if j in plot_trees:
        plt.plot(Ms,test_data[:,j-2],"%s"%types[j-2],label="Tree depth: %d"%j)
plt.legend()
plt.subplot(122)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of learners M")
plt.title("Train Error")
plt.xscale("log",base=2)
plt.ylim(0,15)
for j in tree_depths:
    plt.plot(Ms,train_data[:,j-2],"%s"%types[j-2],label="Tree depth: %d"%j)
plt.legend()
plt.tight_layout()
plt.savefig("../figures/boosting1.pdf")
plt.show()
