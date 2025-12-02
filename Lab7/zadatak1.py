import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def fconv(x,h):
    N = len(x) + len(h) - 1
    xf = np.fft.fft(x,N)
    hf = np.fft.fft(h,N)
    return np.fft.ifft(xf * hf).real

def overlapandadd(x,h,n):
    NN = len(x) + len(h) - 1
    temp = np.zeros(NN)
    for i in range(0, len(x), n):
        temp_y = fconv(x[i : i + n],h) 
        temp[i:i + len(temp_y)] += temp_y
    temp_yend = fconv(x[i + n :],h)
    temp[i + n:] += temp_yend 
    return temp


def saveandselect(x,h,n):
    return np.convolve(x,h)

h = [1,2,1,-1]
x = [1,2,3,1,1,2,3,1]
y = overlapandadd(x,h,4)


fig, [graph1 ,graph2, graph3, graph4] = plt.subplots(nrows = 4, ncols = 1)
graph1.stem(np.arange(len(x)),x)
graph2.stem(np.arange(len(h)),h)
graph3.stem(np.arange(len(y)),fconv(x,h))
graph4.stem(np.arange(len(y)),y)
plt.show()