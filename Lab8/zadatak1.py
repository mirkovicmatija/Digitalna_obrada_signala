import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def myexpand(x, I):
    y = np.zeros(len(x)*I)
    j = 0
    for k in range(0,len(y),I):
        y[k] = x[j]
        j = j + 1
    return y

I = 5
x = [1,2,3,4]
h = myexpand(x,I)

#filtar
for i in range(0,len(h),I):
    for j in range(I):
        h[i+j] = h[i] 

y = h


fig, [graph1 ,graph2, graph3] = plt.subplots(nrows = 3, ncols = 1)
graph1.stem(np.arange(len(x)),x)
graph2.stem(np.arange(len(h)),h)
graph3.stem(np.arange(len(y)),y)
plt.show()
