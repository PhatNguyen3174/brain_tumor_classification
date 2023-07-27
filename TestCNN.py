import cv2
import pickle
from keras.models import load_model
from keras.models import Sequential
from PIL import Image
import numpy as np

model=load_model('BrainTumorCNN.h5')
print(model.summary())

image = cv2.imread('P:\\BrainSVM\\brain_prep\\4 no.jpg')

img = Image.fromarray(image)

img = img.resize((64,64))

img = np.array(img)

print(model.input_shape)
result = model.predict(np.array([img]))
print(np.argmax(result))