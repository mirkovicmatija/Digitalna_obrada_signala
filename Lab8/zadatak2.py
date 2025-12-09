import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



fig, [graph1 ,graph2] = plt.subplots(nrows = 2, ncols = 1)
graph1.stem(np.arange(len(x)),x)
graph2.stem(np.arange(len(h)),h)
plt.show()