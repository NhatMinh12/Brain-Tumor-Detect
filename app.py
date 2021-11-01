from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from keras.models import *
from keras.layers import *
import os

win = Tk()
win.title('app')
win.geometry('700x900+100+50')


def train():
    img_dir = 'Brain Classification'
    dataset = list()
    label = list()
    no_tumor_images = os.listdir(img_dir + '/no_tumor/')
    print(no_tumor_images)
    for i, image_name in enumerate(no_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(f'{img_dir}/no_tumor/{image_name}')
            image = Image.fromarray(image, 'RGB')
            image = image.resize((64, 64))
            dataset.append(np.array(image))
            label.append(0)
    yes_tumor_images = os.listdir(img_dir + '/glioma_tumor/')
    for i, image_name in enumerate(yes_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(f'{img_dir}/glioma_tumor/{image_name}')
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


def get_result(file):
    model = load_model('Brain2.h5')
    img_ = cv2.imread(file)
    img = Image.fromarray(img_)
    img = img.resize((64, 64))
    img = np.array(img)
    input = np.expand_dims(img, axis=0)
    rs = model.predict(input)
    print(rs)
    return int(rs[0][0])


def openDialog():
    global path_file_entry, img_read
    path_file = filedialog.askopenfilename(initialdir="C:/", title="select file",
                                              filetypes=(("image files", "*"), ("all files", "*.*")))
    path_file_entry.insert(0, path_file)
    result = get_result(path_file)
    if result == 1:
        Label(win, text='Tumor detected').place(width=150, height=50, x=20, y=100)
    if result == 0:
        Label(win, text='No tumor').place(width=150, height=50, x=20, y=100)
    img_read = ImageTk.PhotoImage(Image.open(path_file).resize((600, 600), Image.ANTIALIAS))
    Label(win, image=img_read).place(width=600, height=600, x=50, y=250)


Label(win, text='chọn file đáp án').place(width=150, height=50, x=20, y=0)
path_file_entry = Entry(win)
path_file_entry.place(width=400, height=30, x=200, y=10)

Button(win, text='choose', command=openDialog).place(width=70, height=30, x=620, y=10)
win.mainloop()
