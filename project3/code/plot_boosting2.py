import numpy as np
import matplotlib.pyplot as plt
from function_library import *
import pandas as pd
import matplotlib
types=["g","c","k--","r","b--","m--"]
labels=[r"$\lambda_{L1}=1, \eta=0.05$",r"$\lambda_{L1}=1, \eta=0.01$",
        r"$\lambda_{L1}=5, \eta=0.05$",r"$\lambda_{L1}=5, \eta=0.01$",
        r"$\lambda_{L1}=10, \eta=0.05$",r"$\lambda_{L1}=10, \eta=0.01$",]
matplotlib.rcParams.update({"font.size":20})
Ms=np.array(list(range(20,6401,20)),dtype=int) #same as in the files
test_data=np.loadtxt("../csvdata/boosting_2testreduced.csv",
                     skiprows=1,dtype="float",delimiter=",")
train_data=np.loadtxt("../csvdata/boosting_2trainreduced.csv",
                      skiprows=1,dtype="float",delimiter=",")
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of learners M")
plt.title("Test Error")
plt.ylim(0,15)
for j in range(len(labels)):
    plt.plot(Ms,test_data[:,j],"%s"%types[j],label=labels[j])
plt.legend()
plt.subplot(122)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of learners M")
plt.title("Train Error")
plt.ylim(0,15)
for j in range(len(labels)):
    plt.plot(Ms,train_data[:,j],"%s"%types[j],label=labels[j])
plt.legend()
plt.tight_layout()
plt.savefig("../figures/boosting2.pdf")
plt.show()
"""
python3 plot_boosting2.py
