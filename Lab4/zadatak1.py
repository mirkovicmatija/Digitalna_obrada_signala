import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#from scipy.fft import fftshift

fs = 10 ** 7
lt = 10 ** (-4) 
mu = 2.4
t = np.arange(fs*lt)
x = np.cos(2 * np.pi * ((t * 10 ** -6) ** 2) * mu * (10 ** 10))

ff, tt, sxx = signal.spectrogram(x,fs,np.hamming(128),nfft=512,noverlap=8) 

fig, [graph1 ,graph2] = plt.subplots(nrows = 2, ncols = 1)
graph1.plot(t,x)
graph2.pcolormesh(tt, ff, sxx, shading='gouraud')

plt.show()