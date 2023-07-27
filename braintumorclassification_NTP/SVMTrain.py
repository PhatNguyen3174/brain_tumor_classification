import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
import os
import cv2
from sklearn.svm import SVC
from PIL import Image
import pickle
from sklearn import model_selection

image_drec='brain_tumor_dataset/'
dataset =[]
label = []

INPUT_SIZE = 64

no_braintumor = os.listdir(image_drec + 'no/')
yes_braintumor = os.listdir(image_drec + 'yes/')

for i, image_name in enumerate(no_braintumor):
    if(image_name.split('.')[1] =='jpg'):
        image = cv2.imread(image_drec +'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_braintumor):
    if(image_name.split('.')[1] =='jpg'):
        image = cv2.imread(image_drec +'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)
np.unique(label)

dataset = np.reshape(dataset,(len(dataset),-1))

svm = SVC()
xtrain, xtest, ytrain, ytest = train_test_split(dataset,label, random_state = 100, test_size = .20)
xtrain=xtrain/255
xtest=xtest/255
svm.fit(xtrain,ytrain)


filename = 'SVMModel.sav'
pickle.dump(svm, open(filename, 'wb'))













