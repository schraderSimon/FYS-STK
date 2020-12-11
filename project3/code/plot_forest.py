import numpy as np
import matplotlib.pyplot as plt
from function_library import *
import pandas as pd
import matplotlib
matplotlib.rcParams.update({"font.size":20})

n_estimators=np.logspace(0,3.5,8,dtype=int)
Types=["25full","25sqrt","25log","9full","9sqrt","9log","5full","5sqrt","5log"]

test_data=np.loadtxt("../csvdata/randomforest_3.162000e+03test.csv",skiprows=1,dtype="float",delimiter=",")
train_data=np.loadtxt("../csvdata/randomforest_3.162000e+03train.csv",skiprows=1,dtype="float",delimiter=",")
test_25full=test_data[:,0]
test_25sqrt=test_data[:,1]
test_25third=test_data[:,3]
test_9full=test_data[:,4]
test_9sqrt=test_data[:,5]
test_9third=test_data[:,7]
test_5full=test_data[:,8]
train_25full=train_data[:,0]
train_25sqrt=train_data[:,1]
train_25third=test_data[:,3]
train_9full=train_data[:,4]
train_9sqrt=train_data[:,5]
train_9third=test_data[:,7]
train_5full=train_data[:,8]
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of trees")
plt.title("Test Error")
plt.xscale("log")
plt.ylim(0,30)
plt.plot(n_estimators,test_25third,label=r"depth=25, $\frac{all}{3}$predictors")
plt.plot(n_estimators,test_25full,label="depth=25, all predictors")
plt.plot(n_estimators,test_9third,label=r"depth=9, $\frac{all}{3}$predictors")
plt.plot(n_estimators,test_9full,label="depth=9, all predictors")
plt.plot(n_estimators,test_5full,label="depth=5, all predictors")
plt.plot(n_estimators,test_25sqrt,label=r"depth=25, $\sqrt{predictors}$")
plt.plot(n_estimators,test_9sqrt,label=r"depth=9, $\sqrt{predictors}$")
plt.legend()
plt.subplot(122)
plt.ylabel("MAE (kcal/mol)")
plt.xlabel("Number of trees")
plt.title("Train Error")
plt.xscale("log")
plt.ylim(0,30)
plt.plot(n_estimators,train_25third,label=r"depth=25, $\frac{all}{3}$predictors")
plt.plot(n_estimators,train_25full,label="depth=25, all predictors")
plt.plot(n_estimators,train_9third,label=r"depth=9, $\frac{all}{3}$predictors")
plt.plot(n_estimators,train_9full,label="depth=9, all predictors")
plt.plot(n_estimators,train_5full,label="depth=5, all predictors")
plt.plot(n_estimators,train_25sqrt,label=r"depth=25, $\sqrt{predictors}$")
plt.plot(n_estimators,train_9sqrt,label=r"depth=9, $\sqrt{predictors}$")
plt.legend()
print("Minimium 25 full: %f %f"%(test_25full[-1],test_25full[-2]))
print("Minimium 9 full: %f"%test_9full[-1])
print("Minimium 25 third: %f %f"%(test_25third[-1],test_25third[-2]))
print("Minimium 9 third: %f"%test_9third[-1])
plt.tight_layout()
plt.savefig("../figures/forest.pdf")
plt.show()
