import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""

"""
lasso=pd.read_csv("../csvData/Lasso_data.csv")
ridge=pd.read_csv("../csvData/Ridge_data.csv")
OLS=pd.read_csv("../csvData/OLS_data.csv")
print(OLS.head())
test_MSE_OLS=OLS["test_MSE"]
test_MSE_ridge=ridge["test_MSE"]
test_MSE_lasso=lasso["test_MSE"]
maxdeg=max(len(test_MSE_lasso),len(test_MSE_ridge),len(test_MSE_OLS))
plt.xticks(np.arange(0, maxdeg+1, step=1))
plt.title(r"$\sigma$=%f, datapoints: %d"%(OLS["sigma"][0],OLS["datapoints"][0]))
plt.plot(range(1,len(test_MSE_OLS)+1),test_MSE_OLS,label="OLS, BS: %d"%(OLS["n_bootstrap"][0]))
plt.plot(range(1,len(test_MSE_ridge)+1),test_MSE_ridge,label="Ridge, BS: %d"%(ridge["n_bootstrap"][0]))
plt.plot(range(1,len(test_MSE_lasso)+1),test_MSE_lasso,label="LASSO, BS: %d"%(lasso["n_bootstrap"][0]))
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

"""
run as python3 plot_from_data.py
"""
