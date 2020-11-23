import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from function_library import *
from keras.regularizers import l2, l1
import sys
from sklearn.preprocessing import StandardScaler

filedata="../data/qm7.mat"
X,R,Z,T,P=read_data(filedata)
X_train, R_train, Z_train, T_train, X_test, R_test, Z_test, T_test= convert_dataset(X,R,Z,T,P,0)
print(X_train.shape)
T_train=-T_train
T_test=-T_test
#sys.exit(1)
normalizer=preprocessing.Normalization()
X_train_reduced=X_train#X_train[:500,:].copy()
T_train_reduced=T_train#T_train[:500,:].copy()
print(X_train_reduced.shape)
scaler = StandardScaler()
scaler.fit(X_train_reduced)
X_train_reduced_scaled=scaler.transform(X_train_reduced)
X_test_scaled=scaler.transform(X_test)
#normalizer.adapt(X_train_reduced)
"""
first=np.array(X_train[:1])
with np.printoptions(precision=2,suppress=True):
    print("First example",first)
    print("Normalized:", normalizer(first).numpy())
"""
type="sigmoid"

def baseline_model(input,regulizer,learningrate,type):
    model=keras.models.Sequential()
    #model.add(norm)
    model.add(layers.Dense(400,activation=type,kernel_regularizer=regulizer,input_dim=input.shape[1]))

    #model.add(layers.Dense(400,kernel_regularizer=regulizer,kernel_initializer='normal',bias_initializer="random_normal",activation=type))
    model.add(layers.Dense(100,kernel_regularizer=regulizer,kernel_initializer='normal',bias_initializer="random_normal",activation=type))

    model.add(layers.Dense(1))
    optimizer=keras.optimizers.Adam(learning_rate=learningrate)
    model.compile(loss="mean_absolute_error",optimizer=optimizer)
    return model
epochs=int(1e3)
writefile=False
if(writefile):
    outfile=open("test_error_full_%s%d_restricted.csv"%(type,epochs),"w")
    outfile.write("Learning_rate,regression,accuracy\n")
learning_rates=np.logspace(-4,-1,4)#for sigmoid_ -3,-2,3 (with 2 layers a 400, 100)
regulizers_l2=np.logspace(-4.5,-3.5,3)#for sigmoid_ -5,-1,5 (with two layers a 400, 100)
regulizers_l1=np.logspace(-8,-5,4)
for k in learning_rates:

    previous=0
    for i in regulizers_l2:
        #for sigmoid_ -5,-1,5 (with two layers a 400, 100)
        model=baseline_model(X_train_reduced_scaled,l2(i),k,type)
        model.fit(X_train_reduced_scaled, T_train_reduced,epochs=epochs,batch_size=25,verbose=0)

        accuracy=model.evaluate(X_test_scaled,T_test,verbose=0)
        if(previous>10):
            if(accuracy-previous)> 3:
                break
        print("%e %e l2" %(k,i))
        print("Test accuracy: %f"%accuracy)
        print("Train accuracy: %f"%model.evaluate(X_train_reduced_scaled,T_train_reduced,verbose=0))
        if(writefile):
            outfile.write("%e,%e,%f,l2\n"%(k,i,accuracy))
        previous=accuracy
    previous=0
    for i in regulizers_l1:
        model=baseline_model(X_train_reduced_scaled,l1(i),k,type)
        model.fit(X_train_reduced_scaled, T_train_reduced,epochs=epochs,batch_size=25,verbose=0)
        accuracy=model.evaluate(X_test_scaled,T_test,verbose=0)
        if(previous>10):
            if(accuracy-previous)> 3:
                break
        print("%e %e l1" %(k,i))
        print("Test accuracy: %f"%accuracy)
        print("Train accuracy: %f"%model.evaluate(X_train_reduced_scaled,T_train_reduced,verbose=0))
        if(writefile):
            outfile.write("%e,%e,%f,l2\n"%(k,i,accuracy))
        previous=accuracy
if(writefile):
    outfile.close()
