import numpy as np
import matplotlib.pyplot as plt


def laplacian_matrix(N):
    """
    Build a discrete 2D Laplacian on an N x N grid of interior points.
    Output: A matrix of size (N*N) x (N*N).
    """
    size = N * N
    A = np.zeros((size, size))

    # Helper to convert 2D -> 1D index
    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            A[k, k] = -4  # center

            # neighbors with boundary checking
            if i > 0:
                A[k, idx(i - 1, j)] = 1
            if i < N - 1:
                A[k, idx(i + 1, j)] = 1
            if j > 0:
                A[k, idx(i, j - 1)] = 1
            if j < N - 1:
                A[k, idx(i, j + 1)] = 1

    return A


def visualize_mode(vec, N, title):
    """
    Turn eigenvector back into NxN grid and plot it.
    """
    mode = vec.reshape(N, N)
    plt.figure()
    plt.imshow(mode, cmap='coolwarm', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.show()


def main():
    N = 5  # 5x5 grid of interior points
    print("Building Laplacian matrix for a", N, "x", N, "grid...")
    
    A = laplacian_matrix(N)
    print("Matrix A shape:", A.shape)

    # Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(A)

    # Sort by eigenvalue (ascending)
    idx_sort = np.argsort(vals)
    vals = vals[idx_sort]
    vecs = vecs[:, idx_sort]

    print("First 5 eigenvalues of the Laplacian:")
    print(vals[:5])

    # Visualize the first 3 vibration modes
    for mode_number in range(3):
        visualize_mode(vecs[:, mode_number], N,
                       f"Mode {mode_number+1} (Eigenvalue = {vals[mode_number]:.2f})")


if __name__ == "__main__":
    main()
