import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import preprocess as preprocess
import analyse as analyse

#stereo -> mono 
samplerate, data = preprocess.load_audio_mono('test_audio/13-10-22 shotgun.wav')
samplerate_2, data_2 = preprocess.load_audio_mono('test_audio/13-10-22 shotgun 2.wav')

#data spliting
data_array = preprocess.split_audio(data, samplerate, segment_duration=3)
data_array_2 = preprocess.split_audio(data_2, samplerate_2, segment_duration=3)

#applyiing window function and resampling to 16kHz
filtered = preprocess.preprocess_segments(data_array, samplerate, target_samplerate=16000)
filtered_2 = preprocess.preprocess_segments(data_array_2, samplerate_2, target_samplerate=16000)


#feature extraction and z-score normalization
z_array = []
for f in filtered:
    # Compute Mel-spectrogram
    log_mel_spectrogram = analyse.analyze_segments(f, samplerate=16000)
    z = analyse.zscore_normalization(log_mel_spectrogram)
    z_array.append(z)
    #analyse.plot_spectrogram(log_mel_spectrogram)

for f in filtered_2:
    # Compute Mel-spectrogram
    log_mel_spectrogram = analyse.analyze_segments(f, samplerate=16000)
    z = analyse.zscore_normalization(log_mel_spectrogram)
    z_array.append(z)
    #analyse.plot_spectrogram(log_mel_spectrogram)

#labels for training data
train_data = np.array(z_array)
label = np.array([1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,
                  1,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,
                  1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0])

#reshaping data for training
X = preprocess.reshape_segment(train_data)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the classifier
accuracy = knn.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")


# Make predictions
y_pred = knn.predict(X_test)
print(f"Predictions: {y_pred}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
