

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


cov_mat = np.array([[1,   0.8],
                    [0.8,   1]])

mean = np.array([0, 0])

# grid
x = np.linspace(-3, 3, 40)
y = np.linspace(-3, 3, 40)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Gaussian 2D pdf
rv = multivariate_normal(mean, cov_mat)
Z = rv.pdf(pos)

# Plot 2D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_axis_off()


# Gaussian 1D pdf
rv = multivariate_normal(0, 0.7)
z = rv.pdf(x)

# Plot 1D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(x, z)
ax.set_axis_off()

# plot 1D


plt.show()