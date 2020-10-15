import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread
from sklearn import datasets
from function_library import *
from sklearn.neural_network import MLPClassifier
def sigmoid(x):
    return 1/(1+np.exp(-x))
np.random.seed(0)
plt.rcParams['figure.figsize'] = (12,12)


# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))


# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)


def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

#Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)


eta=3e-3
epochs=1000
n_hidden_neurons=[100,100]
n_hidden_layers=len(n_hidden_neurons)
n_categories=10
batch_size=32
Lambda=0.01
activation_function_type="RELU"
activation_function_type_output="softmax"
dnn=NeuralNetwork(X_train,Y_train_onehot,
    n_hidden_layers=n_hidden_layers,n_hidden_neurons=n_hidden_neurons,
    n_categories=n_categories,epochs=epochs,batch_size=batch_size,eta=eta,lmbd=Lambda,
    activation_function_type=activation_function_type,
    activation_function_type_output=activation_function_type_output,
    errortype="categorical")


#dnn=NeuralNetwork(X_train,Y_train_onehot,epochs=100, batch_size=100,eta=0.01,lmbd=0.01,n_categories=10,n_hidden_neurons=50)
dnn.train()
test_predict=dnn.predict(X_test)
print(test_predict)
def accuracy_score(Y_test, y_pred):
    return np.sum(Y_test==y_pred)/len(Y_test)
print("Accuracy score: %f"%accuracy_score(Y_test,test_predict))
clf = MLPClassifier(max_iter=epochs,activation='relu',solver="sgd",hidden_layer_sizes=n_hidden_neurons,learning_rate_init=eta).fit(X_train, Y_train)
y_pred=clf.predict(X_test)
print("MLF: %f"%accuracy_score(Y_test,y_pred))
