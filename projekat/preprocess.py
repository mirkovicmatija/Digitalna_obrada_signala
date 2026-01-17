import numpy as np
from scipy import signal
from scipy.io import wavfile
import json


#stereo -> mono
def load_audio_mono(file_path):
    samplerate, data = wavfile.read(file_path)
    if len(data.shape) == 1:
        return samplerate, data
    data_mono = data[:,1]/2 + data[:,0]/2 
    return samplerate, data_mono

#data spliting
def split_audio(data_mono, samplerate, segment_duration=5):
    data_array = np.array_split(data_mono, round(len(data_mono)/ samplerate / segment_duration))
    return data_array

#applyiing window function and resampling to 16kHz
def preprocess_segments(data_array, original_samplerate, target_samplerate=16000):
    filtered = []
    for d in data_array:
        t = d * np.hamming(len(d))
        smpl = round(len(d) / original_samplerate) * target_samplerate
        a = signal.resample(t, smpl)
        filtered.append(a)
    return filtered

#reshaping data for training
def reshape_segment(train_data):
    return train_data.reshape(train_data.shape[0], -1)

# Help function to write data to JSON file
def data_writing(segment, label, filepath='data.json'):
   
    with open(filepath, 'r') as file:
        data = json.load(file)
    if label == 1:
        data['hornet'].append(segment.tolist())
    elif label == 0:
        data['non_hornet'].append(segment.tolist())
    with open(filepath, 'w') as file:
        json.dump(data, file)