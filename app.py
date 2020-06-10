# dependencies
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import librosa
import os
import math
from pydub import AudioSegment
from tensorflow.keras import load_model

# create the flask app
app = Flask(__name__)

# the base route
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        file = request.files['file']
        name, ext = os.path.splitext(file)
        sound = AudioSegment.from_mp3(file)
        # make 30s clip 
        if len(sound) < 30000:
            duration = 30000-len(sound)
            segment = AudioSegment.silent(duration = duration)
            extract =  sound + segment
        else:
            extract = sound[0:30000]
    
        extract.export(f"{name}.wav", format="wav")
    
        new_file = f"{name}.wav"
        sample_rate = 22050
        n_mfcc = 26
        n_fft = 2048
        hop_length = 512
        num_segments = 1
        duration = 30
        samples_per_track = sample_rate * duration 
        num_samples_per_segment = int(samples_per_track / num_segments)
        expected_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)        
    
        signal, sr = librosa.load(new_file, sr = sample_rate)

        mfcc_list = []

        # for s in range(num_segments):
        #     start_sample = num_samples_per_segment * s
        #     finish_sample = start_sample + num_samples_per_segment
        mfccs = librosa.feature.mfcc(signal,
                                    sr = sr, n_fft = n_fft, 
                                    n_mfcc = n_mfcc, hop_length = hop_length)
        mfccs = mfccs.T
        if len(mfccs) == expected_vectors_per_segment: 
            mfcc_list.append(np.array(mfccs))
        
        data = np.array(mfcc_list)
        X = data[..., np.newaxis]
        bird_model = load_model('cnn_66_trained.h5')
        prediction_array = bird_model.predict(X)
        predicted_index = np.argmax(prediction_array, axis=1)

        species_df = pd.read_csv('unique_species.csv')
    
        for index, rows in species_df.iterrows():
            if index == predicted_index:
                sp = species_df.loc[index, 'Common Name']
                cn = species_df.loc[index, 'Species']
                prediction = (sp, cn)

        return render_template('index.html', prediction = prediction)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)