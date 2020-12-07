import numpy as np
import matplotlib.pyplot as plt
from function_library import *
filedata="../data/qm7.mat"
X,R,Z,T,P=read_data(filedata)
X_removed=reduce_coulomb(create_hydrogenfree_coulomb_matrix(X))
X=reduce_coulomb(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
scaler = StandardScaler()
scaler.fit(X_removed)
X_removed=scaler.transform(X_removed)
pca = PCA()
pca.fit(X)
X=pca.transform(X)
cumulative_varianceX=np.cumsum(pca.explained_variance_ratio_)
pca = PCA()
pca.fit(X_removed)
X_removed=pca.transform(X_removed)
cumulative_varianceXremoved=np.cumsum(pca.explained_variance_ratio_)
cumulative_varianceXremoved=np.pad(cumulative_varianceXremoved, ((1,0)), mode='constant', constant_values=0)
cumulative_varianceX=np.pad(cumulative_varianceX, (1,0), mode='constant', constant_values=0)

plt.plot(np.linspace(0,1,len(cumulative_varianceX)),cumulative_varianceX,label="Reduced Coulomb matrix")
plt.plot(np.linspace(0,1,len(cumulative_varianceXremoved)),cumulative_varianceXremoved,label="Reduced Coulomb matrix without hydrogen")
plt.axhline(0.99,ls="--",color="black",label="0.99")
plt.axhline(0.95,ls="--",color="red",label="0.95")

plt.legend()
plt.title("Cumulative explained variance")
plt.xlabel("Percentage of used Principal Components")
plt.ylabel("Cumulative explained variance")
plt.savefig("../figures/Cumulative_variance.pdf")
plt.show()
