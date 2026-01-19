import numpy as np
from scipy import stats
import librosa
import preprocess as preprocess
import matplotlib.pyplot as plt
from io import BytesIO

# Compute Mel-spectrogram
def analyze_segments(filtered_segments, samplerate=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y=filtered_segments, sr=samplerate, n_mels=128, fmax=8000)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def zscore_normalization(log_mel_spectrogram):
    return stats.zscore(log_mel_spectrogram, axis=1, ddof=1)

#Help function to convert spectrogram to image
def spectrogram_to_image(log_mel_spectrogram):
  
    buf = BytesIO()
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(log_mel_spectrogram, sr=16000, x_axis='time', y_axis='mel', fmax=8000)
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    import PIL.Image as Image
    image = Image.open(buf)
    return image

# Help function to plot spectrogram
def plot_spectrogram(log_mel_spectrogram):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=16000, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()
    
# Help function to plot waveform
def plot_graph(t):
    plt.plot(np.arange(len(t)), t, label='Audio Signal 1', alpha=0.7, color='blue',  linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal Segment Waveform')
    plt.grid()
    plt.show()
   