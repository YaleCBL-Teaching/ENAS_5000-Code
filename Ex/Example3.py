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