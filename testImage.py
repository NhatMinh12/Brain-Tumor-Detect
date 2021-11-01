import cv2
from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('Brain2.h5')
img_ = cv2.imread('image(75).jpg')
img = Image.fromarray(img_)
img = img.resize((64, 64))
img = np.array(img)
input = np.expand_dims(img, axis=0)
rs = model.predict(input)
print(rs)