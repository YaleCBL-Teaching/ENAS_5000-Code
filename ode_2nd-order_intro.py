#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

omega_0 = 1.0
zeta = 0.2
y0 = 2.0
v0 = 0.0
omega_d = omega_0 * np.sqrt(1 - zeta**2)
A = y0
B = (v0 + zeta * omega_0 * A) / omega_d
t = np.linspace(0, 20, 2000)
y = np.exp(-zeta * omega_0 * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(f"A={A:.2f}, B={B:.4f}")
plt.show()