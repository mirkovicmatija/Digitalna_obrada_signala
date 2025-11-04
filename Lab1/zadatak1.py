import numpy as np
import matplotlib.pyplot as plt


def function(t):
    return np.cos(np.pi * t / 3)


n = np.arange(6)
x = function(n)
y = np.fft.fft(x)

n2 = np.arange(3)
x2 = function(n2)
y2 = np.fft.fft(x2)

fig, [graph1, graph2, graph3, graph4] = plt.subplots(nrows = 4, ncols = 1)
graph1.stem(n,x)
graph2.stem(n,np.abs(y))
graph3.stem(n,y.real)
graph4.stem(n,y.imag)
plt.show()

fig2, [graph5, graph6, graph7, graph8] = plt.subplots(nrows = 4, ncols = 1)
graph5.stem(n2,x2)
graph6.stem(n2,np.abs(y2))
graph7.stem(n2,y2.real)
graph8.stem(n2,y2.imag)
plt.show()