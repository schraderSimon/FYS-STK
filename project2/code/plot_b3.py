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
    method="ADAM"
    activation="logistic"
names=["sigmoid","tanh","ReLU","LeakyReLU"]
test=[]
mapname="../csvData/%db1_%s%s%d%d_epoch100"%(num_layers,activation,method,datapoints,degree)
test.append(pd.read_csv("%s/test.csv"%mapname).to_numpy()[:,1:]/1000)
activation="tanh"; mapname="../csvData/%db1_%s%s%d%d_epoch100"%(num_layers,activation,method,datapoints,degree)
test.append(pd.read_csv("%s/test.csv"%mapname).to_numpy()[:,1:]/1000)
activation="RELU"; mapname="../csvData/%db1_%s%s%d%d_epoch100"%(num_layers,activation,method,datapoints,degree)
test.append(pd.read_csv("%s/test.csv"%mapname).to_numpy()[:,1:]/1000)
activation="LeakyRELU"; mapname="../csvData/%db1_%s%s%d%d_epoch100"%(num_layers,activation,method,datapoints,degree)
test.append(pd.read_csv("%s/test.csv"%mapname).to_numpy()[:,1:]/1000)
infile=open("%s/info.txt"%mapname)
lambda_val=np.array(infile.readline().split(",")[1:-1],dtype="float")
eta_simple=(np.array(infile.readline().split(",")[1:-1],dtype="float"))
plt.figure(figsize=(20,10))
axes=[plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)]
threshold=1e2
for i in range(len(eta_simple)):
    for j in range(len(lambda_val)):
        for test_values in test:
            if test_values.T[i,j]>threshold:
                test_values.T[i,j]=np.nan
for i,test_values in enumerate(test):
    minimum=int(np.nanmin(test_values.T)*1000)
    min_pos=np.nanargmin(test_values.T.ravel())
    min_eta=eta_simple[int(min_pos/len(lambda_val))]
    min_lambda=lambda_val[int(min_pos%len(lambda_val))]
    print("%s gives %d at Eta=%.2e and Lambda=%.2e" %(names[i],minimum,min_eta,min_lambda))
xlabels = ['{:.1e}'.format(x) for x in lambda_val];
ylabels= ['{:.0e}'.format(y) for y in eta_simple];
for i in range(2):
    for j in range(2):
        sns.heatmap(test[2*i+j].T, xticklabels=xlabels, yticklabels=ylabels,annot=True,fmt=".1f", ax=axes[2*i+j], cmap="viridis",cbar_kws={'label': r'Magnitude of $10^3$'})
        axes[2*i+j].set_title("test MSE with %s"%names[2*i+j])
        axes[2*i+j].set_ylabel(r"$\eta$")
        axes[2*i+j].set_xlabel(r"$\lambda$")
plt.tight_layout()
plt.savefig("../figures/Four_activations%d%s%d%d_100.pdf"%(num_layers,method,datapoints,degree))
plt.show()
