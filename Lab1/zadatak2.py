import numpy as np
import matplotlib.pyplot as plt

a = 0.8
n = np.arange(6)
x = 0.8 ** n * np.heaviside(n, 1) 

plt.stem(n, x)
plt.show()
