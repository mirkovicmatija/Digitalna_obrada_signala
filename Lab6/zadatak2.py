import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile 
from numpy.polynomial.polynomial import polyadd

def fconv(x,h):
    N = len(x) + len(h) - 1
    xf = np.fft.fft(x,N)
    hf = np.fft.fft(h,N)
    return np.fft.ifft(xf * hf)

h = [1,2,1,-1]
x = [1,2,3,1]
y = fconv(x,h)

xx = [1,2,1]
t1 = [1,1/2,1/4,1/8,1/18,1/32]
t3 = [1,1,1,1]
y1 = fconv(xx,t1) 
y2 = fconv(xx,t3)
yy = polyadd(y1,y2)

samplerate1, data1 = wavfile.read('handel_mono_11025.wav')
samplerate2, data2 = wavfile.read('impulse_cathedral.wav')
d = fconv(data1,data2)

xr = [1,2,1,1]
yxr = fconv(xr,xr)

fig, [(graph1 ,graph2, graph3), (graph4 ,graph5, graph6), (graph7 ,graph8, graph9), (graph0 ,grapha, graphb), 
(graphc ,graphd, graphe)] = plt.subplots(nrows = 5, ncols = 3)
graph1.stem(np.arange(len(x)),x)
graph2.stem(np.arange(len(h)),h)
graph3.stem(np.arange(len(y)),y)

graph4.stem(np.arange(len(x)),x)
graph5.stem(np.arange(-1,3),h)
graph6.stem(np.arange(-1,len(y)-1),y)

graph7.stem(np.arange(len(xx)),xx)
graph8.stem(np.arange(len(t1)),t1)
graph9.stem(np.arange(len(t3)),t3)

graph0.stem(np.arange(len(y1)),y1)
grapha.stem(np.arange(len(y2)),y2)
graphb.stem(np.arange(len(yy)),yy)



graphc.stem(np.arange(len(xr)),xr)
graphd.stem(np.arange(len(xr)),xr)
graphe.stem(np.arange(len(yxr)),yxr)

fig, [graphd1 ,graphd2, graphdd] = plt.subplots(nrows = 1, ncols = 3)
graphd1.plot(np.arange(len(data1)),data1)
graphd2.plot(np.arange(len(data2)),data2)
graphdd.plot(np.arange(len(d)),d)

plt.show()
