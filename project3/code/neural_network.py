import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from function_library import *
import numpy as np

from keras.regularizers import l2, l1
import sys
tf.config.optimizer.set_jit(True)

np.random.seed(272) #L dies after 272 days. RIP L

def neural_network(input,regulizer,learningrate,type,nn_layers=[100,100]):
    """Create a simple N-layer neural network with given regulizer, learning rate"""

    model=keras.models.Sequential()
    #model.add(norm)
    model.add(layers.Dense(nn_layers[0],activation=type,kernel_regularizer=regulizer,input_dim=input.shape[1]))
    for i in range(1,len(nn_layers)):
        model.add(layers.Dense(nn_layers[i],kernel_regularizer=regulizer,kernel_initializer='normal',bias_initializer="random_normal",activation=type))

    model.add(layers.Dense(1))
    optimizer=keras.optimizers.Adam(learning_rate=learningrate)
    model.compile(loss="mean_absolute_error",optimizer=optimizer)
    return model


filedata="../data/qm7.mat"
input_type=["coulomb","reduced","not_coulomb","noH"][int(sys.argv[1])]

X,R,Z,T,P=read_data(filedata)

amount_eta=4 #number of learning rates to iterate over
amount_lambda=7 #number of regulizers to iterate over

learning_rates=np.logspace(-3,0,amount_eta)
regulizers_l2=np.logspace(-9,-3,amount_lambda)
regulizers_l1=np.logspace(-9,-3,amount_lambda)

nn_layers=[100,100] #two layer nn with 100 layers each

epochs=100 #number of epochs
batch_size=25 #batch size for SGD
number_crossvals=1 # The number of cross validations to perform. 5 to get "actual" results

import os
try:
    os.mkdir(mapname)
except:
    pass

"""Adaptive file name"""
filename="../csvdata/nn_"
filename=filename+input_type
for layer in nn_layers:
    filename+="_%d"%layer
filename+="epochs_%d"%epochs
filename+="number_crossvals_%d"%number_crossvals
filename+="ex.csv"
outfile=open(filename,"w")
outfile.write("Activation_function,regulizer,batch_size,lambda,eta,train_err,test_err\n")
maxcounter=2*len(learning_rates)*(len(regulizers_l1)+len(regulizers_l2))
test_errors=np.zeros(maxcounter)
train_errors=np.zeros(maxcounter)
maxcounter*=number_crossvals

for index in range(number_crossvals):
    counter=0
    X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,index)
    meanie=np.mean(T_train)
    T_train=-T_train; T_train=T_train-meanie #Scale T (reduce mean and make positive)
    T_test=-T_test; T_test=T_test-meanie #Scale T (reduce mean and make positive)
    if input_type=="not_coulomb": #The Z-R-approachs
        X_train=np.concatenate((R_train,Z_train),axis=1)
        X_test=np.concatenate((R_test,Z_test),axis=1)
    if input_type=="reduced": #Reduced Coulomb matrix
        pass
    if input_type=="noH": #No Hydrogen
        testing_indeces=P[index]
        training_indeces=np.delete(P,index,0).ravel()
        X_removed=create_hydrogenfree_coulomb_matrix(X)
        X_removed=reduce_coulomb(X_removed)
        X_train, X_test= createTestTrain(X_removed,training_indeces,testing_indeces)
    if input_type=="coulomb": #Data as it is
        testing_indeces=P[index]
        training_indeces=np.delete(P,index,0).ravel()
        Xnew= X.reshape(len(X),-1)
        X_train, X_test= createTestTrain(Xnew,training_indeces,testing_indeces)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)


    for type in ["sigmoid","elu"]:
        for learningrate in learning_rates:
            for l1_reg in regulizers_l1:
                model=neural_network(X_train_scaled,l1(l1_reg),learningrate,type)
                model.fit(X_train_scaled, T_train,epochs=epochs,batch_size=batch_size,verbose=0) #no output
                test_accuracy=model.evaluate(X_test_scaled,T_test,verbose=0)
                train_acuracy=model.evaluate(X_train_scaled,T_train,verbose=0)
                test_errors[counter]+=test_accuracy
                train_errors[counter]+=train_acuracy
                counter+=1
        for learningrate in learning_rates:
            for l2_reg in regulizers_l2:
                model=neural_network(X_train_scaled,l2(l2_reg),learningrate,type)
                model.fit(X_train_scaled, T_train,epochs=epochs,batch_size=batch_size,verbose=0)
                test_accuracy=model.evaluate(X_test_scaled,T_test,verbose=0)
                train_acuracy=model.evaluate(X_train_scaled,T_train,verbose=0)
                test_errors[counter]+=test_accuracy
                train_errors[counter]+=train_acuracy
                counter+=1
counter=0
train_errors/=number_crossvals
test_errors/=number_crossvals
for type in ["sigmoid","elu"]:
    for learningrate in learning_rates:
        for l1_reg in regulizers_l1:
            outfile.write("%s,%s,%d,%e,%e,%f,%f\n"%(type,"l1",batch_size,l1_reg,learningrate,train_errors[counter],test_errors[counter]))
            counter+=1
    for learningrate in learning_rates:
        for l2_reg in regulizers_l2:
            outfile.write("%s,%s,%d,%e,%e,%f,%f\n"%(type,"l2",batch_size,l2_reg,learningrate,train_errors[counter],test_errors[counter]))
            counter+=1
outfile.close()
"""
run as python3 neural_network.py NUMBER
where Number is an integer: 0 - coulomb, 1 - reduced coulomb, 2- R/Z, 3 - reduced no H
"""
