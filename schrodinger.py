import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Parameters
N = 100  # Number of grid points
L = 1.0  # Length of the potential well
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Mass of the particle

# Discretize the domain
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Define the potential (infinite potential well)
V = np.zeros(N)
V[0] = V[-1] = np.inf  # Infinite potential at boundaries

# Construct the finite difference matrix
A = np.zeros((N, N))

# Fill the matrix using the finite difference approximation
for i in range(1, N - 1):
    A[i, i - 1] = 1 / dx**2
    A[i, i] = -2 / dx**2 + V[i]
    A[i, i + 1] = 1 / dx**2

# Apply boundary conditions (wavefunction is zero at walls)
A[0, 0] = A[N - 1, N - 1] = 1

# Solve for eigenvalues and eigenvectors
energies, wavefunctions = eigh(A)

# Normalize wavefunctions
for i in range(len(wavefunctions)):
    wavefunctions[:, i] /= np.sqrt(np.trapz(wavefunctions[:, i]**2, x))

# Plotting
plt.figure(figsize=(12, 8))

# Plot the first few wavefunctions
for i in range(3):
    plt.plot(x, wavefunctions[:, i], label=f'n={i+1}, E={energies[i]:.2f}')
    
plt.title('Wavefunctions for an Infinite Potential Well')
plt.xlabel('Position (x)')
plt.ylabel('Wavefunction (ψ)')
plt.legend()
plt.grid()
plt.show()