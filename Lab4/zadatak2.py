import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

n = np.arange(256)
x = np.cos(0.15 * np.pi * n)
x[79] = 10

windowsFunc = [np.hamming(8),np.hamming(32),np.hamming(64),np.hamming(128)]
ff1, tt1, sxx1 = signal.spectrogram(x,1,windowsFunc[0],nfft=512,noverlap=1)
ff2, tt2, sxx2 = signal.spectrogram(x,1,windowsFunc[1],nfft=512,noverlap=1) 
ff3, tt3, sxx3 = signal.spectrogram(x,1,windowsFunc[2],nfft=512,noverlap=1) 
ff4, tt4, sxx4 = signal.spectrogram(x,1,windowsFunc[3],nfft=512,noverlap=1)  
fig, [graph1 ,graph2, graph3 ,graph4] = plt.subplots(nrows = 4, ncols = 1)
graph1.pcolormesh(tt1, ff1, sxx1, shading='gouraud')
graph2.pcolormesh(tt2, ff2, sxx2, shading='gouraud')
graph3.pcolormesh(tt3, ff3, sxx3, shading='gouraud')
graph4.pcolormesh(tt4, ff4, sxx4, shading='gouraud')
plt.show()

