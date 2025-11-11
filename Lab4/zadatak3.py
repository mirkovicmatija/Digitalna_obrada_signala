import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile 

samplerate, data = wavfile.read('iaeao.wav')

"""
data_x = []

for d in data:
    if(d):
        data_x.append(d)

"""
for i in range(4):
    ff, tt, sxx = signal.spectrogram(data,samplerate,np.hamming(64 * (2 ** i)),nfft=1024,noverlap=31) 
    fig, [graph1 ,graph2] = plt.subplots(nrows = 2, ncols = 1)
    graph1.plot(np.arange(len(data)),data)
    graph2.pcolormesh(tt, ff, sxx, shading='gouraud')
    plt.show()