# This file contains helper functions for the flask app

# dependencies
import pandas as pd
import numpy as np
import librosa
import os
from pydub import AudioSegment
import keras.models
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
 
# use model to predict species from audio sample, deliver predicted label
def make_prediction(data):
    bird_model = load_model('birds2_model.h5')
    data_array = np.array(data)
    X = data_array[..., np.newaxis]
    prediction_array = bird_model.predict(X)

    predicted_index = np.argmax(prediction_array, axis=1)

    species_df = pd.read_csv('bird_sightings.csv')

    species_list = []
    common_names = []
    for row in species_df:
        if row['Species'] in species_list:
            pass
        else:
            species_list.append(row['Species'])
        if row['English_name'] in common_names:
            pass
        else:
            common_names.append(row['English_name'])

    d = {'Species': species_list, 'Common Name': common_names}
    name_df = pd.DataFrame(d)
    name_df = name_df.sort_values('Species')
    name_df = name_df.reset_index(drop=True)
    
    for index, rows in name_df.iterrows():
        if int(predicted_index) == int(index):
            result = (f'{row['Species']}, {row['Common Name']} ')
        break
        
    return result

# load audio file, transform, extract MFCCs
def transform_file(file):
    sample_rate = 22050
    n_mfcc = 26
    n_fft = 2048
    hop_length = 512
    num_segments = 1
    duration = 30
    samples_per_track = sample_rate * duration 
    num_samples_per_segment = int(samples_per_track / num_segments)
    expected_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    signal, sr = librosa.load(file, sr = sample_rate)
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment
        mfccs = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    sr = sample_rate, n_fft = n_fft, 
                                    n_mfcc = n_mfcc, hop_length = hop_length)
        mfccs = mfccs.T
        if len(mfccs) == expected_vectors_per_segment: 

            return mfccs.to_list()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        name, ext = os.path.splitext(file)
        if ext == 'wav':
            sound = AudioSegment.from_wav(file)
            if len(sound) < 30000:
                duration = 30000-len(sound)
                segment = AudioSegment.silent(duration = duration)
                extract =  sound + segment
            else:
                extract = sound[0:30000]
            wav_file = extract.export('data.wav', format = 'wav')
            data = transform_file(wav_file)
        else:
            sound = AudioSegment.from_mp3(file)
            if len(sound) < 30000:
                    duration = 30000-len(sound)
                    segment = AudioSegment.silent(duration = duration)
                    extract =  sound + segment
                else:
                    extract = sound[0:30000]
                wav_file = extract.export('data.wav', format = 'wav')
                data = transform_file(wav_file)
        
        prediction = make_prediction(data)

    return prediction