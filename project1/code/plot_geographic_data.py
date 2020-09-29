import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("../csvData/Korea.csv")
print(data.head())
mindeg=data["mindeg"][0];maxdeg=data["maxdeg"][0]
k=data["k"][0]
datapoints=data["datapoints"][0]
n_bootstraps=data["n_bootstrap"][0]
n_bootstraps_lasso=data["n_bootstraps_lasso"][0]

"""Plot Error and R2 score"""
R2_OLS=data["R2_OLS"]; R2_ridge=data["R2_ridge"];R2_lasso=data["R2_lasso"]
MSE_OLS_BOOTSTRAP=data["MSEBOOTOLS"];MSE_RIDGE_BOOTSTRAP=data["MSEBOOTRIDGE"];MSE_LASSO_BOOTSTRAP=data["MSEBOOTLASSO"];
MSEkfoldRIDGE=data["MSEkfoldRIDGE"];MSEkfoldLASSO=data["MSEkfoldLASSO"];MSEkfoldOLS=data["MSEkfoldOLS"]
x_axis=range(mindeg,maxdeg+1)
xticks=np.arange(mindeg-1, maxdeg+1, step=2)
fig, (ax0, ax1) = plt.subplots(ncols=2,figsize=(10, 5))
ax0.set_title(r"datapoints: %d, bootstrap: %d (%d in lasso)"%(datapoints,n_bootstraps,n_bootstraps_lasso))
ax0.set_xticks(xticks)
ax0.set_xlabel("Polynomial degree")
ax0.set_ylabel("MSE")
ax0.set_ylim(0,7.5*1e4)
ax0.plot(x_axis,MSE_OLS_BOOTSTRAP,"r",label="MSE_OLS_BOOTSTRAP")
ax0.plot(np.argmin(MSE_OLS_BOOTSTRAP)+mindeg,MSE_OLS_BOOTSTRAP[np.argmin(MSE_OLS_BOOTSTRAP)],"ro",markersize="5")
ax0.plot(x_axis,MSE_RIDGE_BOOTSTRAP,"g",label="MSE_RIDGE_BOOTSTRAP")
ax0.plot(np.argmin(MSE_RIDGE_BOOTSTRAP)+mindeg,MSE_RIDGE_BOOTSTRAP[np.argmin(MSE_OLS_BOOTSTRAP)],"go",markersize="5")
print("Minimal error OLS: %d"%MSE_OLS_BOOTSTRAP[np.argmin(MSE_OLS_BOOTSTRAP)])
print("Minimal error Ridge: %d"%MSE_RIDGE_BOOTSTRAP[np.argmin(MSE_OLS_BOOTSTRAP)])
ax0.plot(x_axis,MSE_LASSO_BOOTSTRAP,"b",label="MSE_LASSO_BOOTSTRAP")
ax1.set_title(r"datapoints: %d, k: %d"%(datapoints,k))
ax1.set_xticks(xticks)
ax1.set_xlabel("Polynomial degree")
ax1.set_ylabel("MSE")
ax1.set_ylim(0,7.5*1e4)
ax1.plot(x_axis,MSEkfoldOLS,label="MSE_OLS_CrossVal")
ax1.plot(x_axis,MSEkfoldRIDGE,label="MSE_Ridge_CrossVal")
ax1.plot(x_axis,MSEkfoldLASSO,label="MSE_LASSO_CrossVal")
ax0.legend()
ax1.legend()
plt.savefig("../figures/MSE_different_methods_Korea.pdf")
plt.show()

plt.title(r"datapoints: %d, bootstrap: %d (%d in lasso)"%(datapoints,n_bootstraps,n_bootstraps_lasso))
plt.xticks(xticks)
plt.xlabel("Polynomial degree")
plt.ylabel("R2-score")
plt.ylim(0,1)
plt.plot(x_axis,R2_OLS,label="OLS")
plt.plot(x_axis,R2_ridge,label="Ridge")
plt.plot(x_axis,R2_lasso,label="Lasso")
plt.legend()
plt.savefig("../figures/R2_bootstrap_different_methods_Korea.pdf")
plt.show()
"""Plot bias-variance tradeoff"""
variance_OLS=data["variance_0LS"];bias_OLS=data["bias_OLS"]
variance_Ridge=data["variance_Ridge"]
bias_Ridge=data["bias_RIDGE"]
xticks=np.arange(mindeg-1, maxdeg+1, step=2)
fig, (ax1, ax0) = plt.subplots(ncols=2,figsize=(10, 5))
ax0.set_title(r"Ridge datapoints: %d, bootstrap: %d"%(datapoints,n_bootstraps))
ax0.set_xticks(xticks)
ax0.set_xlabel("Polynomial degree")
ax0.set_ylim(0,7.5*1e4)

ax0.plot(x_axis,variance_Ridge,"r",label="Variance")
ax0.plot(x_axis,bias_Ridge,"g",label=r"Bias^2")
ax0.plot(x_axis,MSE_RIDGE_BOOTSTRAP,"b",label="MSE")

ax1.set_title(r"OLS datapoints: %d, bootstrap: %d"%(datapoints,n_bootstraps))
ax1.set_xticks(xticks)
ax1.set_xlabel("Polynomial degree")
ax1.set_ylim(0,7.5*1e4)
ax1.set_xlim(0,23)
ax1.plot(x_axis,variance_OLS,"r",label="Variance")
ax1.plot(x_axis,bias_OLS,"g",label=r"Bias^2")
ax1.plot(x_axis,MSE_OLS_BOOTSTRAP,"b",label="MSE")
ax0.legend()
ax1.legend()
plt.savefig("../figures/Bias_variance_Korea.pdf")
plt.show()
