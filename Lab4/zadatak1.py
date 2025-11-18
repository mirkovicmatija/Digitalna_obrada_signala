import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


mu = [2.4,4.8,1.2]
t = np.arange(0, 10 ** (-4) ,10 ** (-7))
x = [0,0,0]

for i in range(3):
    x[i] = np.cos(2 * np.pi * (t ** 2) * mu[i] * (10 ** 10))

for i in range(3):
    ff, tt, sxx = signal.spectrogram(x[i],10 ** (-7),np.hamming(128),nfft=512,noverlap=8) 
    fig, [graph1 ,graph2] = plt.subplots(nrows = 2, ncols = 1)
    graph1.plot(t,x[i])
    graph2.pcolormesh(tt, ff, sxx, shading='gouraud')
    plt.show()