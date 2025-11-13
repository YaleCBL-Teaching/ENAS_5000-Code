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

# PRACTICE 1:
# Using my_matrix:
# 1) Extract the LAST two rows and LAST two columns using slicing (no hard-coded numbers).
# 2) Call this submatrix "block".
# 3) Print the sum of all elements inside block.
#    (Use a numpy function, not a loop.)

# PRACTICE 2:
# Using the cube array:
# 1) Select the 2nd “slice” along the first dimension (i.e., cube[1]).
# 2) Compute the column sums of that 2D slice using axis=0.
# 3) Compute the Euclidean norm of that column-sum vector.
# 4) Print the final norm.

