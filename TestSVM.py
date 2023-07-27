import cv2
import pickle
from keras.models import load_model
from PIL import Image
import numpy as np

filename = 'SVMModel.sav'
model = pickle.load(open(filename, 'rb'))

#image = cv2.imread('P:\\BrainSVM\\brain_prep\\Y61.jpg')
image = cv2.imread('P:\\Nhom7_Detai10_NguyenTruongPhat_NguyenNgocTuyetNy\\BrainSVM\\brain_prep\\Y13.jpg')

img = Image.fromarray(image)

img = img.resize((64,64))

img = np.array(img)
img = img.reshape(1,-1)/255

result = model.predict(img)
print(result)