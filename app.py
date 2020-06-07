# dependencies
from flask import Flask, render_template, request, redirect
import numpy as np
import keras.models
import os

# declaring constants
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flv', 'mp4', 'wma'}

# create the flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# the base route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)