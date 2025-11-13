import numpy as np

# 2D array (matrix)
my_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(my_matrix)
# shape = (3 rows, 3 columns)
print(my_matrix.shape)

# Slicing (take first 2 rows, first 2 columns)
print(my_matrix[0:2, 0:2])
# [[1 2]
#  [4 5]]

# Sum down the rows (axis=0 means "go down")
print(np.sum(my_matrix, axis=0))   # [12 15 18]

# 3D array example (just to show shape)
cube = np.arange(24).reshape(4, 3, 2)  # shape (4,3,2)
print(cube.shape)
