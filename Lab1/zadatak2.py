import numpy as np
import matplotlib.pyplot as plt

a = 0.8
n = np.arange(10)
x = 0.8 ** n * np.heaviside(n, 1) 

y1 = np.fft.fft(x,5)
y2 = np.fft.fft(x,50)

fig, [graph1, graph2, graph3, graph4, graph5] = plt.subplots(nrows = 5, ncols = 1)
graph1.stem(n,x)
graph2.stem(np.arange(5),np.abs(y1))
graph3.stem(np.arange(50),np.abs(y2))
graph4.stem(np.arange(5),np.fft.ifft(y1).real)
graph5.stem(np.arange(50),np.fft.ifft(y2).real)
plt.show()
