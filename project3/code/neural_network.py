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
print(X_train.shape)
T_train=T_train
T_test=T_test
print(T_train)
#sys.exit(1)
normalizer=preprocessing.Normalization()
X_train_reduced=X_train[:500,:].copy()
T_train_reduced=T_train[:500,:].copy()
print(X_train_reduced.shape)
normalizer.adapt(X_train_reduced)
"""
first=np.array(X_train[:1])
with np.printoptions(precision=2,suppress=True):
    print("First example",first)
    print("Normalized:", normalizer(first).numpy())
"""
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
outfile=open("test_error_relu.csv","w")
outfile.write("Learning_rate,regression,accuracy\n")
for k in np.logspace(-4,-1,4):
    for i in np.logspace(-6,2,9):
        model=baseline_model(normalizer,l2(i),k)
        model.fit(X_train_reduced, T_train_reduced,epochs=epochs,batch_size=25,verbose=0)
        accuracy=model.evaluate(X_test,T_test)
        print(accuracy)
        outfile.write("%f,%f,%f,l2\n"%(k,i,accuracy))
    for i in np.logspace(-6,2,9):
        model=baseline_model(normalizer,l1(i),k)
        model.fit(X_train_reduced, T_train_reduced,epochs=epochs,batch_size=25,verbose=0)
        accuracy=model.evaluate(X_test,T_test)
        outfile.write("%f,%f,%f,l1\n"%(k,i,accuracy))
outfile.close()
