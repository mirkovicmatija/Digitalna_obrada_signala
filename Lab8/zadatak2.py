import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

n = np.arange(200)
x = np.cos(0.04 * np.pi  * n) + np.cos(0.18 * np.pi  * n)
y = x[::4]
h = np.fft.fft(y).real

fig, [graph1 ,graph2,graph3] = plt.subplots(nrows = 3, ncols = 1)
graph1.stem(np.arange(len(x)),x)
graph2.stem(np.arange(len(y)),y)
graph3.stem(np.arange(len(h)),h)
plt.show()