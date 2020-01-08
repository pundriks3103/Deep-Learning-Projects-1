#Importing the Libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from skimage import io
import os
from keras.preprocessing import image

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def entry_page():
    #Jinja template of the webpage
    return render_template('index.html')

@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    #Get image URL as input
    image_url = request.form['image_url']
    image = io.imread(image_url)
    
    test_image = image.load_img('image', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

             
    #Return the model results to the web page
    return render_template('index.html',prediction_text='The image is {} %'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
