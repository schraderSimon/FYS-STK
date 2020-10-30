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
    degree=20
    num_layers=2
    method="sgd"
    activation="logistic"
    scikit=True
mapname="../csvData/%db1_%s%s%d%d"%(num_layers,activation,method,datapoints,degree)
if scikit:
    Scikit_train=np.array(pd.read_csv("%s/Scikit_train.csv"%mapname).to_numpy()[:,1:])/1000
    Scikit_test=pd.read_csv("%s/Scikit_test.csv"%mapname).to_numpy()[:,1:]/1000
train=pd.read_csv("%s/train.csv"%mapname).to_numpy()[:,1:]/1000
test=pd.read_csv("%s/test.csv"%mapname).to_numpy()[:,1:]/1000

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
        if scikit:
            if Scikit_train.T[i,j]>threshold:
                Scikit_train.T[i,j]=np.nan
            if Scikit_test.T[i,j]>threshold:
                Scikit_test.T[i,j]=np.nan
        if train.T[i,j]>threshold:
            train.T[i,j]=np.nan
        if test.T[i,j]>threshold:
            test.T[i,j]=np.nan
if scikit:
    minimum_scikit=int(np.nanmin(Scikit_test.T)*1000)

    min_pos_scikit=np.nanargmin(Scikit_test.T.ravel())
    min_eta_scikit=eta_simple[int(min_pos_scikit/len(lambda_val))]
    min_lambda_scikit=lambda_val[int(min_pos_scikit%len(lambda_val))]
minimum_test=int(np.nanmin(test.T)*1000)
min_pos=np.nanargmin(test.T.ravel())
min_eta=eta_simple[int(min_pos/len(lambda_val))]
min_lambda=lambda_val[int(min_pos%len(lambda_val))]

xlabels = ['{:.1e}'.format(x) for x in lambda_val];
ylabels= ['{:.0e}'.format(y) for y in eta_simple];
ax1.set_title("train MSE Scikit Learn")
ax2.set_title("test MSE Scikit Learn")
ax3.set_title(r"train MSE own NN")
ax4.set_title(r"test MSE own NN")
if scikit:
    g=sns.heatmap(Scikit_train.T, xticklabels=xlabels, yticklabels=ylabels,annot=True,fmt=".1f", ax=ax1, cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
    h=sns.heatmap(Scikit_test.T,xticklabels=xlabels, yticklabels=ylabels, annot=True, fmt=".1f",ax=ax2, cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
i=sns.heatmap(train.T, xticklabels=xlabels, yticklabels=ylabels,annot=True, ax=ax3,fmt=".1f", cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
j= sns.heatmap(test.T,xticklabels=xlabels, yticklabels=ylabels, annot=True, ax=ax4,fmt=".1f", cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
ax1.set_ylabel(r"$\eta$");ax2.set_ylabel(r"$\eta$");ax3.set_ylabel(r"$\eta$");ax4.set_ylabel(r"$\eta$")
ax1.set_xlabel(r"$\lambda$"); ax2.set_xlabel(r"$\lambda$"); ax3.set_xlabel(r"$\lambda$");ax4.set_xlabel(r"$\lambda$");
print("Minimal Test error with Scikit learn: %d at lambda=%.2e, eta=%.2e"%(minimum_scikit,min_lambda_scikit,min_eta_scikit))
print("Minimal Test error with own NN: %d at lambda=%.2e, eta=%.2e"%(minimum_test,min_lambda,min_eta))

plt.tight_layout()
plt.savefig("../figures/scikit_own_%db1_%s%s%d%d.pdf"%(num_layers,activation,method,datapoints,degree))
plt.show()
