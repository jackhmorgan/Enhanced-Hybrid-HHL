import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')

from enhanced_hybrid_hhl import (HHL, 
                                 HybridInversion,
                                 CannonicalInversion,
                                 Lee_preprocessing,
                                 Yalovetsky_preprocessing,
                                 QuantumLinearSystemProblem)
import numpy as np
from qiskit.quantum_info import random_hermitian

from qiskit_aer import AerSimulator

simulator = AerSimulator()

def hermitian_matrix(eigenvalues):
    # Generate a random unitary matrix
    unitary_matrix = np.linalg.qr(np.random.randn(len(eigenvalues), len(eigenvalues)))[0]

    # Construct the Hermitian matrix using the spectral decomposition
    hermitian = np.dot(np.dot(unitary_matrix, np.diag(eigenvalues)), np.conj(unitary_matrix.T))
    
    return hermitian

def MorganExampleQLSP(eigenvalues: list):
    test_eigenvalues = [-1, -5/7, -2/7, 1/4, 1/3, 11/16, 3/4, 1]
    test_A_matrix = hermitian_matrix(eigenvalues=test_eigenvalues)
    eigh = np.linalg.eigh(test_A_matrix)
    eigenvalue_indexes = [np.where(np.isclose(eigh[0], ev)) for ev in eigenvalues]

    relevant_eigenvectors = [eigh[1][index] for index in eigenvalue_indexes]
    b_vector_example = np.zeros(relevant_eigenvectors[0].shape)
    for vector in relevant_eigenvectors:
        b_vector_example += vector
    b_vector_example /= np.linalg.norm(b_vector_example)

    return QuantumLinearSystemProblem(test_A_matrix, b_vector_example)

problem = MorganExampleQLSP([-1, 11/16])

Cannonical_HHL = HHL(get_result_function='get_fidelity_result',
                     eigenvalue_inversion=CannonicalInversion
                     )

fidelity = Cannonical_HHL.estimate(problem=problem, max_eigenvalue=1)
print(fidelity)

y_preprocessing=Lee_preprocessing(num_eval_qubits=4, backend=simulator, max_eigenvalue=1)

Yalovetsky_H_HHL = HHL(get_result_function='get_fidelity_result',
                       pre_processing=y_preprocessing.estimate,
                       eigenvalue_inversion=HybridInversion
                       )

hybrid_fidelity = Yalovetsky_H_HHL.estimate(problem=problem)
print(hybrid_fidelity)

e_preprocessing=Lee_preprocessing(num_eval_qubits=6, backend=simulator, max_eigenvalue=1)

Enhanced_H_HHL = HHL(get_result_function='get_fidelity_result',
                       pre_processing=y_preprocessing.estimate,
                       eigenvalue_inversion=HybridInversion
                       )
enhanced_fidelity = Enhanced_H_HHL.estimate(problem=problem)
print(enhanced_fidelity)