import csv
import os
import cv2
import numpy as np
import random


# loads an image
def loadImage(filename):
    filename = filename.split(os.sep)[-1]
    current_path = 'img/' + filename
    image = cv2.resize(cv2.imread(current_path), (400,300), interpolation = cv2.INTER_AREA)

    return image

def getModifiedImage(image):
    rows, cols = image.shape[0], image.shape[1]
    transx = random.randrange(-2,3)
    transy = random.randrange(-2,3)
    M = np.float32([[1, 0, transx], [0, 1, transy]])
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

""" converts a map to numpy array """

def map2np(m):
    l = list(m)
    lT = np.swapaxes(l, 0, 1)
    return (np.stack(lT[0]), lT[1])

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
n_train = len(X_train)

""" generate 2 randomly modified images for each original image """
for i in range(2):
    Xy_add = map(lambda x: [getModifiedImage(X_train[x]), y_train[x]], range(n_train))
    X_add, y_add = map2np(Xy_add)
    X_train = np.concatenate((X_train, X_add), axis=0)
    y_train = np.append(y_train, y_add)

from keras import models
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, InputLayer

model = Sequential()
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

# to retrain a previous trained model - comment out to start a new one
model = models.load_model('model.h5')

callback = EarlyStopping(monitor='loss', patience=2)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, callbacks = [callback])

print (model.predict(np.array([X_train[0]]), batch_size = 1))

#save the trained model
model.save('model.h5')
