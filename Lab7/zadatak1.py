import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def fconv(x,h):
    N = len(x) + len(h) - 1
    xf = np.fft.fft(x,N)
    hf = np.fft.fft(h,N)
    return np.fft.ifft(xf * hf).real

def dconv(x,h):
    return np.convolve(x,h)

def overlapandadd(x,h,n):
    NN = len(x) + len(h) - 1
    temp = np.zeros(NN)
    for i in range(0, len(x), n):
        temp_y = dconv(x[i : i + n],h) 
        temp[i:i + len(temp_y)] += temp_y
    return temp


def saveandselect(x,h,n):
    NN = len(x) + len(h) - 1
    temp = np.zeros(NN)
    for i in range(0, len(x), n): 
        if len(x) - i > 2 * n:
            temp_y = dconv(x[i : i + n],h)
            temp[i:i + n] = temp_y[i:i + n]
        else:
            temp_y = dconv(x[i :],h)
            temp[i:] = temp_y
    return temp

h = [1,2,1,-1]
x = [1,2,3,1,1,2,3,1]
y = overlapandadd(x,h,4)
z = saveandselect(x,h,4)

print(np.convolve(x,h))
print(fconv(x,h))
print(y)
print(z)

fig, [graph1 ,graph2, graph3, graph4, graph5] = plt.subplots(nrows = 5, ncols = 1)
graph1.stem(np.arange(len(x)),x)
graph2.stem(np.arange(len(h)),h)
graph3.stem(np.arange(len(y)),fconv(x,h))
graph4.stem(np.arange(len(y)),y)
graph5.stem(np.arange(len(z)),z)
plt.show()