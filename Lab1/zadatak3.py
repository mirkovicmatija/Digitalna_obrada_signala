import numpy as np
import matplotlib.pyplot as plt


n = np.arange(50)
for i in range(50):
    if i < 11:
        n[i] = 1
    else:
        n[i] = 0

y = np.fft.fftn(n)
m = np.append(n, np.zeros(50))
z = np.fft.fftn(m)

fig, [graph1, graph2, graph3, graph4] = plt.subplots(nrows = 4, ncols = 1)
graph1.stem(np.arange(50),n)
graph2.stem(np.arange(50),np.abs(y))
graph3.stem(np.arange(100),m)
graph4.stem(np.arange(100),np.abs(z))
plt.show()