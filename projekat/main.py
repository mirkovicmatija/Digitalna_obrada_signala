import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.neighbors import KNeighborsClassifier
import preprocess as preprocess
import analyse as analyse
import json
from sklearn.model_selection import train_test_split

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
    z = np.concatenate(z).reshape(1, -1)
   

    predictions = knn.predict(z)
    print(f"Predictions: {predictions}")

"""    
    # Plot Mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=16000, x_axis='time', y_axis='mel', fmax=2048)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()


   
    n = int(input("Enter segment number: "))
    with open('data.json', 'r') as file:
        data1 = json.load(file)
    with open('data.json', 'w') as file:
        if n == 1:
            data1['hornet'].append(z.tolist())
        if n == 2:
            data1['non_hornet'].append(z.tolist())
        else:
            pass
        json.dump(data1, file, indent=4)
"""