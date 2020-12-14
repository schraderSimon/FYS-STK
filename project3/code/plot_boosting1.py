import numpy as np
import matplotlib.pyplot as plt
from function_library import *
import pandas as pd
import matplotlib
types=["g","c","k--","r","b--","m--","y"]
matplotlib.rcParams.update({"font.size":20})
tree_depths=[2,3,4,5,6,7,8]
plot_trees=tree_depths #in case not all plots should be potted
Ms=[100,200,400,800,1600,3200] #needs to be the same as in the file
type="noH" #noH or reduced
test_data=np.loadtxt("../csvdata/boosting_1test%s.csv"%type,
                     skiprows=1,dtype="float",delimiter=",")
train_data=np.loadtxt("../csvdata/boosting_1train%s.csv"%type,
                      skiprows=1,dtype="float",delimiter=",")
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of learners M")
plt.title("Test Error")
plt.xscale("log",base=2)
plt.ylim(0,50)
for j in tree_depths:
    if j in plot_trees:
        plt.plot(Ms,test_data[:,j-2],"%s"%types[j-2],label="Tree depth: %d"%j)
plt.legend()
plt.subplot(122)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of learners M")
plt.title("Train Error")
plt.xscale("log",base=2)
plt.ylim(0,50)
for j in tree_depths:
    plt.plot(Ms,train_data[:,j-2],"%s"%types[j-2],label="Tree depth: %d"%j)
plt.legend()
plt.tight_layout()
plt.savefig("../figures/boosting1%s.pdf"%type)
plt.show()
"""
python3 plot_boosting1.py
"""
