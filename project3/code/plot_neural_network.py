import numpy as np
import matplotlib.pyplot as plt
from function_library import *
import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.rcParams.update({"font.size":18})
num_learningrate=4
num_lambda=7
epochs=100
coulomb=np.loadtxt("../csvdata/results_coulomb_100_100epochs_100number_crossvals_1.csv",
                            skiprows=1,dtype="float",delimiter=",",usecols=6)
noH=np.loadtxt("../csvdata/results_noH_100_100epochs_100number_crossvals_1.csv",
                            skiprows=1,dtype="float",delimiter=",",usecols=6)
not_coulomb=np.loadtxt("../csvdata/results_not_coulomb_100_100epochs_100number_crossvals_1.csv",
                            skiprows=1,dtype="float",delimiter=",",usecols=6)
reduced=np.loadtxt("../csvdata/results_reduced_100_100epochs_100number_crossvals_1.csv",
                            skiprows=1,dtype="float",delimiter=",",usecols=6)
amount_eta=4
amount_lambda=7
learning_rates=np.logspace(-3,0,amount_eta)
regulizers_l2=np.logspace(-9,-3,amount_lambda)
regulizers_l1=np.logspace(-9,-3,amount_lambda)
coulombs=reshapey(coulomb,amount_lambda)
reduceds=reshapey(reduced,amount_lambda)
noHs=reshapey(noH,amount_lambda)
not_coulombs=reshapey(not_coulomb,amount_lambda)
types=["sigmoid L1","sigmoid L2"," elu L1"," elu L2"]
for i, type in enumerate(types):
    print(type)
    print("Minimum coulomb: %f"%np.amin(coulombs[i]))
    print("Minimum noH: %f"%np.amin(noHs[i]))
    print("Minimum reduced: %f"%np.amin(reduceds[i]))
    print("Minimum not_coulomb: %f"%np.amin(not_coulombs[i]))

xlabels = ['{:.1e}'.format(x) for x in regulizers_l2];
ylabels= ['{:.1e}'.format(y) for y in learning_rates];
for i, type in enumerate(types):
    plt.figure(figsize=(20,10))
    ax1=plt.subplot(221)
    ax2=plt.subplot(222)
    ax3=plt.subplot(223)
    ax4=plt.subplot(224)
    ylabels = ['{:.1e}'.format(x) for x in regulizers_l2];
    xlabels= ['{:.1e}'.format(y) for y in learning_rates];
    ax1.set_title(r"MAE, reduced coulomb, %s"%type)
    ax2.set_title(r"MAE, coulomb, %s"%type)
    ax3.set_title(r"MAE, no H, %s"%type)
    ax4.set_title("MAE, Z/R, %s"%type)
    g1=sns.heatmap(reduceds[i].T, xticklabels=xlabels, yticklabels=ylabels,annot=True, ax=ax1,fmt=".1f", cmap="viridis")#,cbar_kws={'label': r'Magnitude of $10^3$'})
    h1=sns.heatmap(coulombs[i].T, xticklabels=xlabels, yticklabels=ylabels,annot=True, ax=ax2,fmt=".1f", cmap="viridis")#,cbar_kws={'label': r'Magnitude of $10^3$'})
    i1=sns.heatmap(noHs[i].T, xticklabels=xlabels, yticklabels=ylabels,annot=True, ax=ax3,fmt=".1f", cmap="viridis")#,cbar_kws={'label': r'Magnitude of $10^3$'})
    j1=sns.heatmap(not_coulombs[i].T,xticklabels=xlabels, yticklabels=ylabels, annot=True, ax=ax4,fmt=".1f", cmap="viridis")#,cbar_kws={'label': r'Magnitude of $10^3$'})
    ax1.set_ylabel(r"$\gamma$");ax2.set_ylabel(r"$\gamma$");ax3.set_ylabel(r"$\gamma$");ax4.set_ylabel(r"$\gamma$")
    ax1.set_xlabel(r"$\lambda$"); ax2.set_xlabel(r"$\lambda$"); ax3.set_xlabel(r"$\lambda$");ax4.set_xlabel(r"$\lambda$");
    plt.tight_layout()
    plt.savefig("../figures/nn_%d_%s.pdf"%(epochs,type))
    plt.show()
