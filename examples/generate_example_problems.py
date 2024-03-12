import numpy as np
from scipy.stats import ortho_group
import json

import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')
from enhanced_hybrid_hhl import QuantumLinearSystemProblem

test_eigenvalues = [-21/24, -20/24, -19/24, -18/24, -17/24, -16/24, -15/24, -14/24, 
                    8/24, 9/24, 10/24, 11/24, 12/24, 13/24, 14/24, 14/24]



def hermitian_matrix(eigenvalues):
    n = len(eigenvalues)
    
    # Generate a random real-valued matrix
    random_matrix = np.random.rand(n, n)
    
    # Make the matrix Hermitian
    hermitian_matrix = random_matrix + random_matrix.T
    
    # Diagonalize the matrix to get eigenvalues
    eigenvals, eigenvectors = np.linalg.eigh(hermitian_matrix)
    
    # Adjust the diagonal elements to match the desired eigenvalues
    adjusted_matrix = np.diag(eigenvalues) + np.dot(np.dot(eigenvectors, np.diag(eigenvalues - eigenvals)), eigenvectors.T)
    
    return adjusted_matrix

def hermitian_matrix2(eigenvalues):
    n = len(eigenvalues)
    ortho = ortho_group.rvs(dim=n)
    diag = np.diag(eigenvalues)
    matrix = np.matmul(ortho, diag)
    matrix = np.matmul(matrix, ortho.T)
    return matrix

def equal_superposition_eigenvectors(eigenvalue_list, matrix):
    # Compute the eigenvectors of the matrix
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Initialize the superposition vector
    superposition_vector = np.zeros_like(eigenvectors[:,0], dtype=complex)
    
    # Iterate over each eigenvalue and add its eigenvector to the superposition vector
    for eigenvalue in eigenvalue_list:
        index = np.abs(eigenvalues - eigenvalue).argmin()
        superposition_vector += eigenvectors[:, index]
    
    # Normalize the superposition vector
    superposition_vector /= np.linalg.norm(superposition_vector)
    
    return superposition_vector

A_matrix = hermitian_matrix2(test_eigenvalues)

relevant_eigenvalues = [-18/24, 8/24]

b_vector = equal_superposition_eigenvectors(relevant_eigenvalues, A_matrix).astype(np.float64)

problem = QuantumLinearSystemProblem(A_matrix, b_vector)

with open('examples/example_problem_16.json', 'w') as f:
    json.dump(problem.__json__(), f)