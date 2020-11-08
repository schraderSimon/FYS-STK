import numpy as np
import matplotlib.pyplot as plt

from function_library import *
import pandas as pd
import sys
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({"font.size":18})
try:
    datapoints=sys.argv[1]
    degree=sys.argv[2]
    num_layers=sys.argv[3]
    method=sys.argv[4]
    activation=sys.argv[5]
except:
    datapoints=2000
    degree=10
    num_layers=2
    method="sgd"
    activation="logistic"
method="ADAM"
mapname="../csvData/%db1_%s%s%d%d"%(num_layers,activation,method,datapoints,degree)
train_ADAM=pd.read_csv("%s/train.csv"%mapname).to_numpy()[:,1:]/1000
test_ADAM=pd.read_csv("%s/test.csv"%mapname).to_numpy()[:,1:]/1000
method="RMSProp"
mapname="../csvData/%db1_%s%s%d%d"%(num_layers,activation,method,datapoints,degree)
train_RMSProp=pd.read_csv("%s/train.csv"%mapname).to_numpy()[:,1:]/1000
test_RMSProp=pd.read_csv("%s/test.csv"%mapname).to_numpy()[:,1:]/1000
infile=open("%s/info.txt"%mapname)
lambda_val=np.array(infile.readline().split(",")[1:-1],dtype="float")
eta_simple=(np.array(infile.readline().split(",")[1:-1],dtype="float"))
plt.figure(figsize=(20,10))
ax1=plt.subplot(221)
ax2=plt.subplot(222)
ax3=plt.subplot(223)
ax4=plt.subplot(224)
threshold=1e2
for i in range(len(eta_simple)):
    for j in range(len(lambda_val)):
        if train_ADAM.T[i,j]>threshold:
            train_ADAM.T[i,j]=np.nan
        if test_RMSProp.T[i,j]>threshold:
            test_RMSProp.T[i,j]=np.nan
        if train_RMSProp.T[i,j]>threshold:
            train_RMSProp.T[i,j]=np.nan
        if test_ADAM.T[i,j]>threshold:
            test_ADAM.T[i,j]=np.nan
minimum_ADAM=int(np.nanmin(test_ADAM.T)*1000)
min_pos_ADAM=np.nanargmin(test_ADAM.T.ravel())
min_eta_ADAM=eta_simple[int(min_pos_ADAM/len(lambda_val))]
min_lambda_ADAM=lambda_val[int(minimum_ADAM%len(lambda_val))]
minimum_RMSProp=int(np.nanmin(test_RMSProp.T)*1000)
min_pos_RMSProp=np.nanargmin(test_RMSProp.T.ravel())
min_eta_RMSProp=eta_simple[int(min_pos_RMSProp/len(lambda_val))]
min_lambda_RMSProp=lambda_val[int(min_pos_RMSProp%len(lambda_val))]
xlabels = ['{:.1e}'.format(x) for x in lambda_val];
ylabels= ['{:.0e}'.format(y) for y in eta_simple];
ax1.set_title("train MSE with ADAM")
ax2.set_title("test MSE with ADAM")
ax3.set_title(r"train MSE with RMSProp")
ax4.set_title(r"test MSE with RMSProp")
g=sns.heatmap(train_ADAM.T, xticklabels=xlabels, yticklabels=ylabels,annot=True,fmt=".1f", ax=ax1, cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
h=sns.heatmap(test_ADAM.T,xticklabels=xlabels, yticklabels=ylabels, annot=True, fmt=".1f",ax=ax2, cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
i=sns.heatmap(train_RMSProp.T, xticklabels=xlabels, yticklabels=ylabels,annot=True, ax=ax3,fmt=".1f", cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
j= sns.heatmap(test_RMSProp.T,xticklabels=xlabels, yticklabels=ylabels, annot=True, ax=ax4,fmt=".1f", cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
ax1.set_ylabel(r"$\eta$");ax2.set_ylabel(r"$\eta$");ax3.set_ylabel(r"$\eta$");ax4.set_ylabel(r"$\eta$")
ax1.set_xlabel(r"$\lambda$"); ax2.set_xlabel(r"$\lambda$"); ax3.set_xlabel(r"$\lambda$");ax4.set_xlabel(r"$\lambda$");
print("Minimal Test error with ADAM: %d at lambda=%.2e, eta=%.2e"%(minimum_ADAM,min_lambda_ADAM,min_eta_ADAM))
print("Minimal Test error with RMSProp: %d at lambda=%.2e, eta=%.2e"%(minimum_RMSProp,min_lambda_RMSProp,min_eta_RMSProp))

plt.tight_layout()
plt.savefig("../figures/ADAM_RMSProp_%db1_%s%s%d%d.pdf"%(num_layers,activation,method,datapoints,degree))
plt.show()

"""
python3 plot_b2.py
"""
