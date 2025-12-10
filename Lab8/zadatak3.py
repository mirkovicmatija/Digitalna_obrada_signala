import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Original discrete data
x_discrete = np.array([0, 1, 2, 3, 4])
y_discrete = np.array([1, 3, 2, 4, 0])

# Create a zero-order hold interpolator
f_zoh = interp1d(x_discrete, y_discrete, kind="zero")

# Generate new x-values for interpolation
x_interp = np.linspace(0, 4, 100)

# Perform zero-order hold interpolation
y_interp_zoh = f_zoh(x_interp)

# Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(x_discrete, y_discrete, 'o', label='Original Samples')
plt.plot(x_interp, y_interp_zoh, '-', label='Zero-Order Hold Interpolation')
plt.title('Zero-Order Hold Interpolation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()