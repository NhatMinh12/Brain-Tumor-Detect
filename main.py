import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from keras.models import *
from keras.layers import *

img_dir = 'archive'
dataset = list()
label = list()
no_tumor_images = os.listdir(img_dir + '/no/')
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(f'{img_dir}/no/{image_name}')
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(0)
yes_tumor_images = os.listdir(img_dir + '/yes/')
for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(f'{img_dir}/yes/{image_name}')
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = utils.normalize(x_train, axis=1)
x_test = utils.normalize(x_test, axis=1)

# y_train = utils.to_categorical(y_train, num_classes=2)
# y_test = utils.to_categorical(y_test, num_classes=2)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
# model.add(Dense(2))
model.add(Activation('sigmoid'))
# model.add(Activation('softmax'))

model.compile(
    loss='binary_crossentropy',
    # loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(x_train, y_train,
          batch_size=16, verbose=1, epochs=10,
          validation_data=(x_test, y_test), shuffle=False
          )
model.save('Brain2.h5')
