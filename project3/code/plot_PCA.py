import pandas as pd
from function_library import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
X_ridge_data=pd.read_csv("../csvdata/ridgereduced.csv")
Xremoved_ridge_data=pd.read_csv("../csvdata/ridgenoH.csv")
X_test=X_ridge_data["test_errors"]
X_train=X_ridge_data["train_errors"]
X_PCA=X_ridge_data["PCA_values"]
Xremoved_test=Xremoved_ridge_data["test_errors"]
Xremoved_train=Xremoved_ridge_data["train_errors"]
Xremoved_PCA=Xremoved_ridge_data["PCA_values"]
fig1, ax1 = plt.subplots()
ax1.set_yscale('log')
ax1.set_yticks(np.logspace(1,2.2,8))
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_ylabel("MAE (kcal/mol)")
ax1.set_xlabel("Number of Principal Components")
ax1.set_title("MAE using Ridge Regression")
ax1.plot(Xremoved_PCA,Xremoved_train,label="train, no H")
ax1.plot(Xremoved_PCA,Xremoved_test,label="test, no H")
ax1.plot(X_PCA,X_train,label="train, with H")
ax1.plot(X_PCA,X_test,label="test, with H")
ax1.legend()

plt.savefig("../figures/Ridge_error.pdf")
plt.show()
"""
python3 plot_PCA.py
"""
