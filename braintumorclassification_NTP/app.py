import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pickle


app = Flask(__name__)

filename = 'SVMModel.sav'
model = pickle.load(open(filename, 'rb'))


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Brain Tumor"
	

def getResult(img):
    image=cv2.imread(img)
    img = Image.fromarray(image)
    img = img.resize((64,64))
    img = np.array(img)
    img = img.reshape(1,-1)/255
    result = model.predict(img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
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
    app.run(debug=True)