import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display


# return a new mnist dataset w/ only the required classes
def disjoint_mnist(mnist,classes):
    pos_train = []
    for i in range(len(classes)):
        tmp = np.where(mnist.train.labels == classes[i])[0]
        pos_train = np.hstack((pos_train,tmp))
        pos_train = np.asarray(pos_train).astype(int)
        np.random.shuffle(pos_train)
    pos_validation = []
    for i in range(len(classes)):
        tmp = np.where(mnist.validation.labels == classes[i])[0]
        pos_validation = np.hstack((pos_validation,tmp))
        pos_validation = np.asarray(pos_validation).astype(int)
        np.random.shuffle(pos_validation)
    pos_test = []
    for i in range(len(classes)):
        tmp = np.where(mnist.test.labels == classes[i])[0]
        pos_test = np.hstack((pos_test,tmp))
        pos_test = np.asarray(pos_test).astype(int)
        np.random.shuffle(pos_test)
    pos=[]
    pos.append(pos_train)
    pos.append(pos_validation)
    pos.append(pos_test)
    
    mnist2 = lambda:0
    mnist2.train = lambda:0
    mnist2.validation = lambda:0
    mnist2.test = lambda:0
    
    mnist2.train.images = mnist.train.images[pos[0]]
    mnist2.validation.images = mnist.validation.images[pos[1]]
    mnist2.test.images = mnist.test.images[pos[2]]
    mnist2.train.labels = mnist.train.labels[pos[0]]
    mnist2.validation.labels = mnist.validation.labels[pos[1]]
    mnist2.test.labels = mnist.test.labels[pos[2]]
    
    return mnist2


# load MNIST dataset with added padding
def load_mnist_32x32(verbose=True):
    # mnist data is by default 28x28 so we add a padding to make it 32x32
    data = input_data.read_data_sets('MNIST_data', one_hot=False, reshape=False)
    # data cannot be directly modified because it has no set() attribute,
    # so we need to make a copy of it on other variables
    X_trn, y_trn = data.train.images, data.train.labels
    X_val, y_val = data.validation.images, data.validation.labels
    X_tst, y_tst = data.test.images, data.test.labels
    # we make sure that the sizes are correct
    assert(len(X_trn) == len(y_trn))
    assert(len(X_val) == len(y_val))
    assert(len(X_tst) == len(y_tst))
    # print info
    if verbose:
        print("Training Set:   {} samples".format(len(X_trn)))
        print("Validation Set: {} samples".format(len(X_val)))
        print("Test Set:       {} samples".format(len(X_tst)))
        print("Labels: {}".format(y_trn))
        print("Original Image Shape: {}".format(X_trn[0].shape))
    # Pad images with 0s
    X_trn = np.pad(X_trn, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_val = np.pad(X_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_tst = np.pad(X_tst, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    if verbose:
        print("Updated Image Shape: {}".format(X_trn[0].shape))
    
    # this is a trick to create an empty object,
    # which is shorter than creating a Class with a pass and so on...
    mnist = lambda:0
    mnist.train = lambda:0
    mnist.validation = lambda:0
    mnist.test = lambda:0
    # and we remake the structure as the original one
    mnist.train.images = X_trn
    mnist.validation.images = X_val
    mnist.test.images = X_tst
    mnist.train.labels = y_trn
    mnist.validation.labels = y_val
    mnist.test.labels = y_tst
    
    return mnist

