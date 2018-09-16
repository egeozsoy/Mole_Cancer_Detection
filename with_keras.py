import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from neural_network import randomize_two_lists

images = np.load('images.npy')
images = np.divide(images, 255)  # normalize
labels = np.load('labels.npy')
images, labels = randomize_two_lists(images, labels)
images = images.reshape(-1, 64 ,64,1)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=images.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(images, labels,
          batch_size=32,
          epochs=10,
          validation_split=0.1)