import numpy as np
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import preprocess as preprocess
import analyse as analyse

# hornet audio preprocessing
with open('data.json', 'r') as file:
        d = json.load(file)
        hornets = d['hornet']
        no_hornet = d['non_hornet']


train_data = np.array(hornets + no_hornet)
label = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0])

#reshaping data for training
X = preprocess.reshape_segment(train_data)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2) # Choose a value for k
knn.fit(X_train, y_train)

# Evaluate the classifier
accuracy = knn.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Testing the classifier

#stereo -> mono 
samplerate, data = preprocess.load_audio_mono('test_audio/13-10-22 shotgun.wav')

#data spliting
data_array = preprocess.split_audio(data, samplerate, segment_duration=5)

#applyiing window function and resampling to 16kHz
filtered = preprocess.preprocess_segments(data_array, samplerate, target_samplerate=16000)


for f in filtered:
    # Compute Mel-spectrogram
    log_mel_spectrogram = analyse.analyze_segments(f, samplerate=16000)
    z = analyse.zscore_normalization(log_mel_spectrogram)
    #analyse.plot_spectrogram(log_mel_spectrogram)
    z = np.concatenate(z).reshape(1, -1)
    #preprocess.data_writing(z, label=-1, filepath='test_data.json')
    predictions = knn.predict(z)
    print(f"Predictions: {predictions}")

