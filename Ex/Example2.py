import numpy as np

# A NumPy array (fast, same-type data)
salaries = np.array([50_000, 60_000, 70_000])

# Basic processing
print(salaries.mean())      # average
print(salaries * 1.05)      # raise everyone 5%

# Famous NumPy functions
print(np.min(salaries))     # smallest value
print(np.max(salaries))     # largest value
print(np.sum(salaries))     # sum of all values
print(np.sort(salaries))    # sorted array
print(np.arange(5))         # array: [0 1 2 3 4]
print(np.linspace(0, 1, 5)) # 5 points from 0 to 1

# PRACTICE 1: Using the salaries array, print the average of the smallest and largest salary.


# PRACTICE 2: Create a new array of salaries after a 10% raise, and then print the difference between the highest raised salary and the lowest raised salary.
