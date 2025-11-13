import numpy as np
import matplotlib.pyplot as plt


def build_1d_conduction_matrix(N, alpha):
    """
    Build the 1D conduction operator for:
        dT/dt = alpha * T_xx
    on the interval (0,1) with T(0)=T(1)=0.
    Using N interior points.
    """
    h = 1.0 / (N + 1)
    A = np.zeros((N, N))

    # Fill matrix using 1D second-order finite difference
    for i in range(N):
        A[i, i] = -2
        if i > 0:
            A[i, i - 1] = 1
        if i < N - 1:
            A[i, i + 1] = 1

    return alpha / h**2 * A


def plot_mode(vec, title):
    plt.figure()
    plt.plot(vec, marker='o')
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    N = 20      # number of interior points
    alpha = 1.0

    # Build discrete conduction operator
    A = build_1d_conduction_matrix(N, alpha)

    # Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(A)

    # Sort by eigenvalue (smallest eigenvalue first)
    idxs = np.argsort(vals)
    vals = vals[idxs]
    vecs = vecs[:, idxs]

    print("First 5 eigenvalues:", vals[:5])

    # Plot first three modes
    for m in range(3):
        plot_mode(vecs[:, m], f"1D Thermal Mode {m+1} (λ = {vals[m]:.2f})")


if __name__ == "__main__":
    main()


    # PRACTICE 1:
    # Using the matrix A:
    # 1) Compute the sum of each ROW of A and store it in an array called row_sums.
    #    (Hint: use np.sum with axis=1.)
    # 2) Print the row_sums array.
    # 3) Find which row has the LARGEST row sum (you can use a simple loop),
    #    and print:
    #       - the index of that row
    #       - the value of its row sum.


 # PRACTICE 2:
    # Using the first three eigenvectors (columns of vecs):
    # 1) For each of the first three modes, compute the sum of its entries
    #    and the mean of its entries (use np.sum and np.mean).
    # 2) Print these values for each mode.
    # 3) Then, for the FIRST mode only, use np.diff to compute the difference
    #    between consecutive entries of the eigenvector and print the maximum
    #    of these differences using np.max.


