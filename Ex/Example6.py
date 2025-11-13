import numpy as np
import matplotlib.pyplot as plt

# 1) Make a 2D grid (x, y) in [0, 1] × [0, 1]
N = 30
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)  # X and Y are both N×N arrays

# 2) Define a "temperature" field T(x, y)
#    Here: a warm spot in the center that decays outward
T = np.exp(-20 * ((X - 0.5)**2 + (Y - 0.5)**2))

# 3) Compute numerical gradient of T with respect to x and y
dx = x[1] - x[0]
dy = y[1] - y[0]
dT_dy, dT_dx = np.gradient(T, dy, dx)  # note: first axis is y, second is x

# 4) Plot temperature field
plt.figure()
plt.imshow(T, origin="lower",
           extent=[x.min(), x.max(), y.min(), y.max()],
           cmap="viridis")
plt.colorbar(label="Temperature")
plt.title("2D Temperature Field")
plt.xlabel("x")
plt.ylabel("y")

# 5) Plot gradient arrows on top (how temperature changes in space)
plt.quiver(X, Y, dT_dx, dT_dy, color="white", scale=30)
plt.show()

# PRACTICE 1:
# Using the temperature field T:
# 1) Compute the average temperature of ONLY the inner region of the grid:
#       (remove the first and last rows AND the first and last columns)
#    Hint: use slicing like T[1:-1, 1:-1].
# 2) Compute the average temperature of the entire grid using np.mean(T).
# 3) Print the difference between these two averages.


# PRACTICE 2:
# Using dT_dx and dT_dy:
# 1) Create a new array "grad_strength" defined as:
#          grad_strength = |dT_dx| + |dT_dy|
#    (Use elementwise absolute value: multiply by +1 or -1 depending on sign.)
# 2) Compute the sum of all elements in grad_strength (use np.sum).
# 3) Compute the mean of grad_strength (use np.mean).
# 4) Print both results.

