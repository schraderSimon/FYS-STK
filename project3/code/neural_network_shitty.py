import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from function_library import *
from keras.regularizers import l2, l1
import sys
filedata="../data/qm7.mat"
X,R,Z,T,P=read_data(filedata)
X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,0)
T_train=T_train
T_test=T_test
print(T_train)
#sys.exit(1)
training_input=np.concatenate((R_train,Z_train),axis=1)
testing_input=np.concatenate((R_test,Z_test),axis=1)
print(R_train.shape,Z_train.shape, training_input.shape)
normalizer=preprocessing.Normalization()
X_train_reduced=training_input[:500,:].copy()
T_train_reduced=T_train[:500,:].copy()
print(X_train_reduced.shape)
normalizer.adapt(X_train_reduced)
def baseline_model(norm,regulizer,learningrate):
    model=keras.models.Sequential()
    model.add(norm)
    #model.add(layers.Dense(400,activation="relu"))
    #model.add(layers.Dense(100,activation="relu"))
    model.add(layers.Dense(400,kernel_regularizer=regulizer,kernel_initializer='normal',bias_initializer="random_normal",activation="relu"))
    model.add(layers.Dense(100,kernel_regularizer=regulizer,kernel_initializer='normal',bias_initializer="random_normal",activation="relu"))

    model.add(layers.Dense(1))
    optimizer=keras.optimizers.Adam(learning_rate=learningrate)
    model.compile(loss="mean_absolute_error",optimizer=optimizer)
    return model
epochs=int(1e3)
#outfile=open("test_error_elu.csv","w")
#outfile.write("Learning_rate,regression,accuracy\n")
for k in np.logspace(-3,-1,5):
    #for sigmoid: -3,-1,5
    previous=0
    for i in np.logspace(-9,-2,8):
        #for sigmoid: -9,-2,8
        model=baseline_model(normalizer,l2(i),k)
        model.fit(X_train_reduced, T_train_reduced,epochs=epochs,batch_size=25,verbose=0)
        accuracy=model.evaluate(testing_input,T_test,verbose=0)
        if(previous>10):
            if(accuracy-previous)> 3:
                break
        print("%s %f %f l2"%(k,i,accuracy))
        #outfile.write("%f,%f,%f,l2\n"%(k,i,accuracy))
        previous=accuracy
    previous=0
    for i in np.logspace(-9,-2,8):
        model=baseline_model(normalizer,l1(i),k)
        model.fit(X_train_reduced, T_train_reduced,epochs=epochs,batch_size=25,verbose=0)
        accuracy=model.evaluate(testing_input,T_test,verbose=0)
        if(previous>10):
            if(accuracy-previous)> 3:
                break
        #print("%f %f %f l1"%(k,i,accuracy))
        #outfile.write("%f,%f,%f,l1\n"%(k,i,accuracy))
        previous=accuracy
#outfile.close()
