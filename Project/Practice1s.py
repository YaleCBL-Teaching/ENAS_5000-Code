import numpy as np
import matplotlib.pyplot as plt

#Input parameters: 

#The length of the domain in x and y direction
Lx = 4.0
Ly = 4.0
#Drichlet Boundary condition values
dbc = 3
#Neumann Boundary condition values
nbc = 5

# ---------------------------
# 1. Grid setup
# ---------------------------

def laplace(Nx, Ny):
    #create grid points in x and y direction
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    
    #calculate grid spacing (delta x and delta y) 
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    # Flattened system size
    N = Nx * Ny

    # ---------------------------
    # 2. Allocate A and b
    # ---------------------------
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Helper: convert (i,j) → k (to be able to solve the system of equations with the correct indexing)
    def idx(i, j):
        return i + j * Nx

    # ---------------------------
    # 3. Fill A and b
    # ---------------------------
    for j in range(Ny):
     for i in range(Nx):
        k = idx(i, j)
        
        # 1) left boundary → Neumann
        if i == 0:
            # (u_1,j - u_0,j)/hx = q
            A[k, idx(1, j)] =  1.0
            A[k, idx(0, j)] = -1.0 
            b[k]            = nbc*hx  # or q_N(y[j])
        
        # 2) right / top / bottom → Dirichlet
        elif i == Nx-1 or j == 0 or j == Ny-1:
            A[k, k] = 1.0
            b[k]    = dbc  # or u_D(x[i], y[j])
        
        # 3) interior
        else:
            A[k, k]              = -4.0
            A[k, idx(i+1, j)]    =  1.0
            A[k, idx(i-1, j)]    =  1.0
            A[k, idx(i, j+1)]    =  1.0
            A[k, idx(i, j-1)]    =  1.0
            b[k] = 0
        print(A)

    # ---------------------------
    # 4. Solve A T = b
    # ---------------------------
    return x, y, np.linalg.solve(A, b)



# Convert back to 2D (This loop help you refine your mesh and develop a convergence study)
n_converge = 5

for n in range(n_converge):
    N = 2**(n+2)
    x, y, T = laplace(N, N)

    T = T.reshape((N, N))

    # ---------------------------
    # 5. Plot result
    # ---------------------------
    X, Y = np.meshgrid(x, y)

    plt.figure()
    plt.contourf(X, Y, T, levels=5)
    plt.colorbar()
    plt.title("2D Laplace Equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

