import numpy as np
import matplotlib.pyplot as plt

from function_library import *
import pandas as pd
import sys
try:
    datapoints=int(sys.argv[1])
    degree=int(sys.argv[2])
    batchsize=int(sys.argv[3])
    num_etas=int(sys.argv[4])
    epochs=int(sys.argv[5])

except:
    datapoints=2000
    degree=10
    batchsize=16
    num_etas=10#array length for etas & t1_values
    epochs=1000
OLS_error_eta=pd.read_csv("../csvData/OLSMSE_eta_datap%ddeg%dbatchs%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))
OLS_error_epochs=pd.read_csv("../csvData/OLSMSE_epochs_datap%ddeg%dbatchs%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))
OLS_error_batchsize=pd.read_csv("../csvData/OLSMSE_batches_datap%ddeg%dbatches%detas%depochs%d.csv"%(datapoints,degree,batchsize,num_etas,epochs))
anal=OLS_error_eta["analytical_error"][0]
plt.figure(figsize=(10,10))
ax1=plt.subplot(221)
ax2=plt.subplot(222)
ax3=plt.subplot(223)
ax4=plt.subplot(224)
ax1.set_title(r"MSE($\eta$)")
ax2.set_title(r"MSE($\eta$)")
ax3.set_title(r"MSE(epochs)")
ax4.set_title(r"MSE(batch size)")
eta_1=np.array(OLS_error_eta["eta"])
ax1.plot([eta_1[0],eta_1[-1]],[anal,anal],label="analytical")
ax1.plot(eta_1,np.array(OLS_error_eta["MSE_ADAM"]),label="ADAM")
ax1.plot(eta_1,np.array(OLS_error_eta["MSE_RMSprop"]),label="RMSProp")
ax1.plot(eta_1,np.array(OLS_error_eta["MSE_SGD"]),label="SGD")

ax1.legend()
ax1.set_xlabel(r"Learning rate $\eta$")
ax1.set_ylabel(r"MSE")
ax1.set_xscale("log")
ax1.set_ylim(0.9*anal,1e5)
print("1. datapoints: %d, degree: %d, batchsize: %d, epochs: %d"%(datapoints,degree,batchsize,epochs))
ax2.plot([np.array(OLS_error_eta["t1"])[0],np.array(OLS_error_eta["t1"])[-1]],[anal,anal],label="analytical")

ax2.plot(np.array(OLS_error_eta["t1"]),np.array(OLS_error_eta["MSE_decay"]),label=r"$t_0$ = %d"%OLS_error_eta["t0"][0])

ax2.set_xscale("log")
ax2.set_ylim(0.9*anal,1e5)
ax2.set_xlabel(r"$t_1$")
ax2.set_ylabel(r"MSE")
ax2.legend()
epochs=np.array(OLS_error_epochs["epochs"])
ax3.plot([epochs[0],epochs[-1]],[anal,anal],label="analytical")
ax3.plot(epochs,np.array(OLS_error_epochs["MSE_ADAM"]),label="ADAM")
ax3.plot(epochs,np.array(OLS_error_epochs["MSE_RMSprop"]),label="RMSProp")
ax3.plot(epochs,np.array(OLS_error_epochs["MSE_SGD"]),label="SGD")
ax3.plot(epochs,np.array(OLS_error_epochs["MSE_decay"]),label="decay")
ax3.legend()
ax3.set_xlabel(r"Number of epochs")
ax3.set_ylabel(r"MSE")
ax3.set_xscale("log")
ax3.set_ylim(0.9*anal,1e5)

batch_sizes=np.array(OLS_error_batchsize["batchsize"])
ax4.plot([batch_sizes[0],batch_sizes[-1]],[anal,anal],label="analytical")
ax4.plot(batch_sizes,np.array(OLS_error_batchsize["MSE_ADAM"]),label="ADAM")
ax4.plot(batch_sizes,np.array(OLS_error_batchsize["MSE_RMSprop"]),label="RMSProp")
ax4.plot(batch_sizes,np.array(OLS_error_batchsize["MSE_SGD"]),label="SGD")
#ax4.plot(batch_sizes,np.array(OLS_error_batchsize["MSE_decay"]),label="decay")
ax4.legend()
ax4.set_xlabel(r"Number of batches")
ax4.set_ylabel(r"MSE")
ax4.set_xscale("log")
#ax4.set_ylim(0.9*anal,1e5)
plt.tight_layout()
plt.savefig("../figures/OLS_error_SGD.pdf")
plt.show()
