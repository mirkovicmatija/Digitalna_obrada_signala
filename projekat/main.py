import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import librosa
import preprocess as preprocess

# hornet audio preprocessing

#no-hornet audio preprocessing





#stereo -> mono 
samplerate, data = preprocess.load_audio_mono('test_audio/13-10-22 shotgun.wav')

#data spliting
data_array = preprocess.split_audio(data, segment_duration=5, target_samplerate=16000)

#applyiing window function and resampling to 16kHz
filtered = preprocess.preprocess_segments(data_array, samplerate, target_samplerate=16000)


for f in filtered:
    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=f, sr=16000, n_mels=128, fmax=8000)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    print(f"mel shape: {log_mel_spectrogram.shape}")
    # Plot Mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=16000, x_axis='time', y_axis='mel', fmax=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()
    # Z-score normalization
    z = stats.zscore(log_mel_spectrogram, axis=1, ddof=1)
    print(z.shape)
        

"""
filtered_array = np.array(filtered)
filtered_list = np.concatenate(filtered_array, axis=None)
print(f"filtered shape: {filtered_list.shape}")


ff, tt, sxx = signal.spectrogram(filtered_list,16000)#,np.hamming(len(f)),nfft=len(f),noverlap=31) 
fig, [graph1, graph2, graph3] = plt.subplots(nrows = 3, ncols = 1)
graph1.plot(np.arange(len(filtered_list)),filtered_list)
graph2.plot(np.arange(len(filtered_list)),np.fft.fft(filtered_list).real)
graph3.pcolormesh(tt, ff, sxx, shading='gouraud')
plt.show()
"""
