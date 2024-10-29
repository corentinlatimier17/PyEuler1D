import numpy as np

class JacobiSolver:
    def __init__(self, tol=1e-10, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self,A, b):
        n = len(b)
        x_old = np.zeros_like(b)  # Initial guess for x
        x_new = np.zeros_like(b)
        
        # Diagonal elements of A
        D = np.diag(A)
        # R = A - D (off-diagonal part of A)
        R = A - np.diagflat(D)
        
        for iteration in range(self.max_iter):
            # Jacobi iteration
            x_new = (b - np.dot(R, x_old)) / D
            
            # Check for convergence (using infinity norm)
            if np.linalg.norm(x_new - x_old, ord=np.inf) < self.tol:
                print(f'Converged after {iteration+1} iterations')
                return x_new[:, 0]
            
            x_old = x_new.copy()

        print(f'Did not converge after {self.max_iter} iterations')
        return x_new[:,0]

class DirectSolver:
    def __init__(self):
        pass
    def solve(self, A, b):
        return np.linalg.solve(A,b)