import numpy as np

# 1) Create an array of x-values
x = np.linspace(0, 10, 6)  # 6 points from 0 to 10

# 3 dimensional vector

vector3 = np.array([1,2,3])

# 2) Vectorized squaring: y = x^2
y = x**2

# 3) Vectorized elementwise multiplication of two arrays
z = x * y  # each element: z[i] = x[i] * y[i]

# 4) Vectorized normalization of x (turn it into a unit vector)
length_vector3 = np.linalg.norm(vector3)  # Euclidean length of x
normalized_unit_vector3 = vector3 / length_vector3         # each element divided by the same scalar

# 5) Approximate gradient of y with respect to x (dy/dx)
dy_dx = np.gradient(y, x)     # numerical derivative on the grid x

print("x       =", x)
print("y = x^2 =", y)
print("z = x*y =", z)
print("||vector3||   =", length_vector3)
print("normalized_unit_vector3  =", normalized_unit_vector3)
# print(np.sqrt(sum(normalized_unit_vector3**2)))
print("dy/dx   =", dy_dx)
