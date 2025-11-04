import numpy as np
import matplotlib.pyplot as plt

def freqplot(x, Fs, N, win):
    res = np.multiply(x,win)
    y = np.abs(np.fft.fft(res,N))
    plt.stem(np.arange(N), y)
    plt.show()



n = np.arange(100)
t = 0.5 * np.cos(2 * np.pi * n / 22 ) + 0.5 * np.cos(2 * np.pi * n / 41 ) 
window_length = 100
blackman_window = np.blackman(window_length)

freqplot(t,100,16,blackman_window)