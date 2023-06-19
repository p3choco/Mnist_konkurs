
from idlelib import history

import tensorflow as tf
from keras import models, datasets
from keras.applications.densenet import layers
from keras.metrics import Recall, Precision
from keras.utils.version_utils import callbacks
from numpy.random import seed
seed(123)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler, History, Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.metrics import precision_score, recall_score
from keras import backend as K
import matplotlib.pyplot as plt

def Zad_1():
    fashion_mnist = datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    n_classes = 10

    X_valid, X_train = np.expand_dims(X_train_full[:5000] / 255., axis=-1), np.expand_dims(X_train_full[5000:] / 255.,
                                                                                           axis=-1)
    X_test = np.expand_dims(X_test / 255., axis=-1)

    history_conv_max_2 = History()

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:], padding="same", activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))
    model.summary()

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_valid = np_utils.to_categorical(y_valid, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    early_stopping = EarlyStopping(patience=10, monitor="val_loss")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy', Precision(), Recall()])
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), validation_split=0.25, epochs=50, callbacks=[early_stopping, history_conv_max_2])

    history_dict = history_conv_max_2.history
    epochs = range(1, len(history_dict['accuracy']) + 1)
    # Wykresy metryk
    plt.figure(figsize=(14, 5))
    print(history_dict.keys())
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history_dict['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, history_dict['loss'], label='Training Loss')
    plt.plot(epochs, history_dict['val_loss'], label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    results = model.evaluate(X_test, y_test)
    print("Loss:", results[0])
    print("Accuracy:", results[1])



Zad_1()


