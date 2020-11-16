import scipy.io as sio

filedata="../data/qm7.mat"

data=sio.loadmat(filedata)
print(data)
print(data.keys())
X_data=data["X"]; #this is the 7165*23*23 array
R_data=data["R"]; #These are the positions: 7165*23*3
Z_data=data["Z"]; #These are the charges: 7165*23
T_data=data["T"]; #This is the atomization energy!! 1*7165
P_data=data["P"]; # This is just a split for 5-fold cross validation 5*1433
# First part: Simple Ridge Regression on X
# Second part a: Neural Network
# Second part b: See how well I can do with only Z & R
# (Second part c): See how well a and b combine
# Third part: Support vector machine or Random Forests
# Fourth part: Perform a friendly remoovy-doovy and repeat a-c
# Fifth part:
