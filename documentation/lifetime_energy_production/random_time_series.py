import numpy as np
from matplotlib import pyplot as plt

def generate_ar1(n, phi, sigma, start=0):
    x = np.zeros(n + 1)
    x[0] = start
    for t in range(1, n + 1):
        x[t] = phi * x[t-1] + sigma * np.random.randn()
    return x


# time
N = 500
dt = 0.1

time = np.arange(0, (N + 1) * dt, dt)

phi_s = 0.9
sigma_s = 0.5
wind_speed = generate_ar1(N, phi_s, sigma_s, start=1)

phi_d = 0.98
sigma_d = 0.5
wind_direction = generate_ar1(N, phi_d, sigma_d)

phi_s = 0.9
sigma_s = 0.5
electricity_price = - wind_speed + generate_ar1(N, phi_s, 0.4 * sigma_s, start=0)


fig, ax1 = plt.subplots()

color = 'red'
ax1.plot(time, wind_speed, color=color)

color = 'green'
ax1.plot(time, wind_direction, color=color)

color = 'blue'
ax1.plot(time, electricity_price, color=color)

plt.show()