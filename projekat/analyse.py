import numpy as np
from scipy import stats
import librosa
import preprocess as preprocess

def analyze_segments(filtered_segments, samplerate=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y=filtered_segments, sr=samplerate, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def zscore_normalization(log_mel_spectrogram):
    return stats.zscore(log_mel_spectrogram, axis=1, ddof=1)