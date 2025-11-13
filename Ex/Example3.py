import numpy as np

# Create an array
my_array = np.array([1, 2, 3])

# Access one element
print(my_array[1])        # 2

# Array properties
print(my_array.dtype)     # int64  (type of elements)
print(my_array.shape)     # (3,)   (array length)
print(my_array.astype(str))  # ['1' '2' '3']

# Generate a sequence
seq = np.arange(1, 10)    # 1 to 9
print(seq)

# Famous NumPy functions
print(np.sum(seq))        # sum of all
# For loops
total = 0
print(len(my_array))
for i in range(0,len(my_array),1):
    total += my_array[i]

print(total)
print(np.max(seq))        # largest
print(np.min(seq))        # smallest
print(np.mean(seq))       # average
print(np.median(seq))     # median
print(np.diff(seq))       # difference between elements

# Broadcasting examples
a = np.array([12, 4, 6, 3, 4, 3, 7, 4])
print(a * 2)              # multiply each element by 2

b = np.array([10, 9, 2, 8, 9, 3, 8, 5])
print(a - b)              # subtract arrays elementwise

#creating a vector

vector = np.array([3,4])
# calculating the euclidean distance

norm = np.linalg.norm(vector)


print(norm)

# PRACTICE 1: Using my_array, compute the sum of its elements WITHOUT using a loop or np.sum().
# (Hint: use simple indexing or slicing.)

# PRACTICE 2: Using the vector array, create a NEW vector [6, 8],
# compute its Euclidean norm, and then print the difference between the two norms.

# PRACTICE 3:
# Using ONLY numpy operations (no loops, no manual +),
# create a new array called "double_seq" that contains seq multiplied by 2.
# Then compute and print ALL of the following:
# 1) The sum of double_seq
# 2) The average (mean) of double_seq
# 3) The difference between consecutive elements of double_seq (use np.diff)

# PRACTICE 4:
# WITHOUT using vstack or new functions:
# 1) Create a new 2D array "M" where the first row is 'a' and the second row is 'b'.
#    (Hint: use np.array([a, b]))
# 2) Compute the column sums of M (axis=0).
# 3) Compute the Euclidean norm of the column-sum vector.
# 4) Subtract the original norm of the vector [3, 4] from this new norm.
# Print the result.



