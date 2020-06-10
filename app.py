# dependencies
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras import models
import os
import process_audio


# declaring constants
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flv', 'mp4', 'wma'}

# create the flask app
app = Flask(__name__)

# the base route
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        prediction = upload_file()
        return render_template('index.html', prediction = prediction)



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)