import numpy as np
import h5py
import matplotlib.pyplot as plt

#Deep learning framework
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Conv2D , Flatten ,MaxPool2D

def create_model():
    model = Sequential()
    model.add(Conv2D(32, input_shape=(64, 64, 3), kernel_size=3,
                 padding = 'same',activation='relu'))
    model.add(MaxPool2D(pool_size = (8,8),strides = (8,8),padding = 'same'))

    model.add(Conv2D(16, kernel_size = 3,activation='relu'))
    model.add(MaxPool2D(pool_size=(4,4),strides = (4,4) ,padding = "same"))
    model.add(Flatten())
    model.add(Dense(6,activation='softmax'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def load_dataset():
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y