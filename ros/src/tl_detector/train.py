from tensorflow.contrib.layers import flatten
import tensorflow as tf
import csv
import os
import cv2
import numpy as np

EPOCHS = 50
BATCH_SIZE = 128

def LeNet(x, n_classes, keep_prob):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    fc0 = tf.nn.dropout(fc0, keep_prob)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


# loads an image
def loadImage(filename):
    filename = filename.split(os.sep)[-1]
    current_path = 'img/' + filename
    image = cv2.resize(cv2.imread(current_path), (400,300), interpolation = cv2.INTER_AREA)

    return image

# read all lines of the driving_log.csv
lines = []
hasFirst = False
with open('labels.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        if not hasFirst:
            hasFirst = True
            continue
        lines.append(line)

images = map(lambda x: loadImage(x[0]), lines)
labels = map(lambda x: int(x[1]), lines)

print (labels)
print (images[0].shape)

X_train = np.array(images)
y_train = np.array(labels)

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout, InputLayer
#from keras.losses import CategoricalCrossentropy

model = Sequential()
#model.add(Cropping2D(cropping=((50,24), (0,0)), input_shape=(160,320,3)))
model.add(InputLayer(input_shape=(300,400,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Dropout(0.2))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(3, activation='softmax'))

# compile and fit
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
callback = EarlyStopping(monitor='loss', patience=2)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, callbacks = [callback])

print (model.predict(np.array([X_train[0]]), batch_size = 1))

#save the trained model
model.save('model.h5')
