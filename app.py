import base64

from flask import Flask, render_template, request
import re
import os
import numpy as np
from keras.models import load_model
from scipy.misc import imread, imresize, imshow
import tensorflow as tf
from train import trainModel

app = Flask(__name__)

global model, graph

def init():
    loaded_model = load_model('model.hdf5')
    print("Loaded Model from disk")

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()

    return loaded_model, graph


# initialize these variables
model, graph = init()


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    # print(imgstr)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    x = x / 255
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    # convert the response to a string
    response = np.array_str(np.argmax(out, axis=1))
    return response


@app.route('/predictUpload/', methods=['GET', 'POST'])
def predictUpload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'upload.png')
    f.save(file_path)
    x = imread('upload.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    x = x / 255
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    # convert the response to a string
    response = np.array_str(np.argmax(out, axis=1))
    return response



@app.route('/train/', methods=['GET', 'POST'])
def train():
    req_data = request.get_json()
    batchsizeinput = req_data['batchsizeinput']
    nbepochinput = req_data['nbepochinput']
    print("batchsizeinput:"+batchsizeinput)
    print("nbepochinput:" + nbepochinput)
    trainModel(int(batchsizeinput), int(nbepochinput))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='localhost', port=port)
