import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

t = np.arange(0, 0.1 , 1/16000.0 )
x = np.cos(np.pi * t * 200)
cwtmatr = signal.cwt(x, signal.ricker, np.arange(1,129))
cwtmatr_yflip = np.flipud(cwtmatr)
xx = np.cos(np.pi * t * 200)
xx[199] = 20
cwtmatrx = signal.cwt(xx, signal.ricker, np.arange(1,129))
cwtmatr_yflipx = np.flipud(cwtmatrx)

tt = np.arange(0, 0.1 , 1/8000.0 )
fun = [np.cos(np.pi * tt * 100), np.cos(np.pi * tt * 200), np.cos(np.pi * tt * 500), np.cos(np.pi * tt * 1000)]
res = np.concatenate((fun[0],fun[1],fun[2],fun[3]))

cwtmatr_r = signal.cwt(res, signal.ricker, np.arange(1,65))
cwtmatr_yflip_r = np.flipud(cwtmatr_r)

cwtmatr_r1 = signal.cwt(res, signal.ricker, np.hamming(16))
cwtmatr_yflip_r1 = np.flipud(cwtmatr_r1)

cwtmatr_r2 = signal.cwt(res, signal.ricker, np.hamming(128))
cwtmatr_yflip_r2 = np.flipud(cwtmatr_r2)

fig, [(graph1 ,graph2), (graph3 ,graph4), (graph5 ,graph6), (graph7 ,graph8)] = plt.subplots(nrows = 4, ncols = 2)
graph1.plot(t,x)
graph2.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
graph3.plot(t,xx)
graph4.imshow(cwtmatr_yflipx, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(cwtmatrx).max(), vmin=-abs(cwtmatrx).max())
graph5.plot(np.arange(0, 0.4 , 1/8000.0 ),res)
graph6.imshow(cwtmatr_yflip_r, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr_r).max(), vmin=-abs(cwtmatr_r).max())
graph7.imshow(cwtmatr_yflip_r1, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr_r1).max(), vmin=-abs(cwtmatr_r1).max())
graph8.imshow(cwtmatr_yflip_r2, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr_r2).max(), vmin=-abs(cwtmatr_r).max())
plt.show()