import numpy as np
import matplotlib.pyplot as plt


n = np.arange(6)
x = np.cos(np.pi * n / 3)

plt.plot(n, x, color='green')
plt.show()
