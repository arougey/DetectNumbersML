import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#each picture is 28x28 pixels with 784 total pixels
#each pixel has a single value from 0-255 inclusive
  #this value determines how dark the pixel is

data = pd.read_csv('/Users/alexrougebec/Projects/DetectAlphabetML/digit-recognizer/train.csv')
#save as numpy array to perform LA and calculations
data = np.array(data)
#get the dimensions of the data
m, n = data.shape

np.random.shuffle(data)

#take the first 1000 data points for the development and transpose so that each picture is a column
data_dev = data[0:1000].T
#get the first picture as Y_dev
Y_dev = data_dev[0]

#get the remaining pictures as X_dev
X_dev = data_dev[1:n]

#take the remaining pictures after 1000 and 
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape

def init_params():
  W1 = np.random.rand(10,784) -.5
  b1 = np.random.rand(10, 1) -.5
  W2 = np.random.rand(10,10) -.5
  b2 = np.random.rand(10, 1) -.5
  return W1, b1, W2, b2

def ReLU(Z):
  return np.maximum(Z, 0)

def softmax(Z):
  return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T #max each column an example
  return one_hot_Y

def deriv_ReLU(Z):
  return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1)
  return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1
  W2 = W2 - alpha * dW2
  b2 = b2 - alpha * db2
  return W1, b1, W2, b2

def get_predictions(A2):
  return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
  print(predictions, Y)
  return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
  W1, b1, W2, b2 = init_params()
  print(X.shape)
  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    if i % 10 == 0:
      print("Iteration", i)
      print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
  return W1, b1, W2, b2
  
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, .1)