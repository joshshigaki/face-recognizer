import os
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

from flask import Flask, request, redirect, url_for, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
graph = None


def load_model():
    global model
    global graph
    # model = Xception(weights="imagenet")
    # model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(48, 48, 3), weights='imagenet')
    model = Sequential()
    model.add(Dense(units=6, activation='relu', input_dim=2))
    model.add(Dense(units=2, activation='softmax'))
    graph = K.get_session().graph

    topLayerModel = Sequential()
topLayerModel.add(Dense(256, input_shape=(512,), activation='relu'))
topLayerModel.add(Dense(256, input_shape=(256,), activation='relu'))
topLayerModel.add(Dropout(0.5))
topLayerModel.add(Dense(128, input_shape=(256,), activation='relu'))
topLayerModel.add(Dense(NUM_CLASSES, activation='softmax'))


load_model()


def prepare_image(img):
    img = raw_data["pixels"][0] # first image
    val = img.split(" ")
    x_pixels = np.array(val, 'float32')
    x_pixels /= 255
    x_reshaped = x.reshape(48,48)
    # return the processed image
    return x_reshaped


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            # Load the saved image using Keras and resize it to the Xception
            # format of 224x224 pixels
            image_size = (48, 48)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size,
                                                    grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)

            global graph
            with graph.as_default():
                preds = model.predict(image)
                results = decode_predictions(preds)
                # print the results
                print(results)

                data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True

        return jsonify(data)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)
