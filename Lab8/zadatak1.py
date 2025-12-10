import numpy as np
import matplotlib.pyplot as plt
import scipy.io

mat = scipy.io.loadmat('BLIdata.mat')


def myexpand(x, I):
    y = np.zeros(len(x)*I)
    j = 0
    for k in range(0,len(y) ,I):
        y[k] = x[j]
        j = j + 1
    return y


def zeroOrder(m,I):
    for i in range(0,len(m),I):
        for j in range(I):
            m[i+j] = m[i] 
    return m

def linearInter(m,I):
    print(len(m) - I,len(m))
    for i in range(0,len(m) - I,I):
        for j in range(I):
            m[i+j] = m[i] + (m[i + I] - m[i]) * j / I
    for i in range(len(m) - I ,len(m)):
        m[i] = m[len(m) - I ]
    return m

I = 5
x = [1,2,3,4]
h = myexpand(x,I)
k = myexpand(x,I)

h = zeroOrder(h,I)
k = linearInter(k,I)

y = np.fft.fft(h).real
z = np.fft.fft(k).real


fig, [graph1 ,graph2, graph3, graph4, graph5] = plt.subplots(nrows = 5, ncols = 1)
graph1.stem(np.arange(len(x)),x)
graph2.stem(np.arange(len(h)),h)
graph3.stem(np.arange(len(y)),y)
graph4.stem(np.arange(len(k)),k)
graph5.stem(np.arange(len(z)),z)
plt.show()
