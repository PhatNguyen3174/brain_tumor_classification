import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pickle


app1 = Flask(__name__)

model=load_model('BrainTumorCNN.h5')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"
	

def getResult(img):
    image=cv2.imread(img)
    img = Image.fromarray(image)
    img = img.resize((64,64))
    img = np.array(img)
    result = model.predict(np.array([img]))
    result = np.argmax(result)
    return result


@app1.route('/', methods=['GET'])
def index():
    return render_template('indexCNN.html')

@app1.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, '/', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None




if __name__ == '__main__':
    app1.run(debug=True)