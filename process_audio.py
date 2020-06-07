# This file contains helper functions for the flask app

# dependencies
import numpy as np
import librosa
import os
from pydub import AudioSegment
import keras.models
from werkzeug.utils import secure_filename

# load and compile model
 

# use model to predict species from audio sample, deliver predicted label


# load audio file, transform, extract MFCCs
def transform_file(file):



    return mfccs

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            name, ext = os.path.splitext(file)
            if ext == 'wav':
                sound = AudioSegment.from_file(file, format=ext)
                if len(sound) < 30000:
                    duration = 30000-len(sound)
                    segment = AudioSegment.silent(duration = duration)
                    extract =  sound + segment
                else:
                    extract = sound[0:30000]
                wav_file = extract.export('data.wav', format = 'wav')
                mfccs = transform_file(wav_file)
            else:
                sound = AudioSegment.from_file(file, format=ext)
                if len(sound) < 30000:
                    duration = 30000-len(sound)
                    segment = AudioSegment.silent(duration = duration)
                    extract =  sound + segment
                else:
                    extract = sound[0:30000]
                wav_file = extract.export('data.wav', format = 'wav')
                mfccs = transform_file(wav_file)