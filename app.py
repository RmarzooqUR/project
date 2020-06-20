import flask
import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf

jsonFile = open('model_arch.json','r')
jsonString=jsonFile.read()
model = tf.keras.models.model_from_json(jsonString)
model.load_weights('model_weights.h5')

model.compile(optimizer = tf.optimizers.Adam(lr = 0.001,beta_1 = 0.9,beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),loss='categorical_crossentropy',metrics=['accuracy'])

def predictLabels(InputImage):
    orderedLabels = ['Actinic keratoses and intraepithelial carcinoma','Basal Cell Carcinoma','Benign keratosis-like lesions','Dermatofibroma','Melanoma','Melanocytic nevi ','Vascular lesions']
    mean = 159.88412604426694
    std = 46.45458064512114
    InputImage = np.expand_dims((np.asarray(InputImage.resize((100,75)))-mean)/std,axis=0)
    return dict(zip(orderedLabels,model.predict(InputImage)[0]))


# ------------------------------------------------------------------------------------------------


app = flask.Flask(__name__,template_folder='templates')

@app.route('/',methods = ['GET', 'POST'])

def main():
    if (flask.request.method == 'GET'):
        return(flask.render_template('main.html'))

    if (flask.request.method == 'POST'):
        imageIn = flask.request.files['file']
        imageIn.save(imageIn.filename)
        pred = predictLabels(Image.open(imageIn.filename))
        os.remove(imageIn.filename)
        return(flask.render_template('main.html',origImage = imageIn,result = pred))


if __name__ == '__main__':
    app.run()
