import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')

from enhanced_hybrid_hhl import (HHL, 
                                 EnhancedHybridInversion,
                                 HybridInversion,
                                 CannonicalInversion,
                                 Lee_preprocessing,
                                 Yalovetsky_preprocessing,
                                 ideal_preprocessing,
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


test_eigenvalues = [-1, -0.6, 0.12222, 1/4, 1/3, 11/32, 3/4, 1]
problem_A_matrix = np.diag(test_eigenvalues)
problem_b_vector = np.array([[0], [1], [0], [0], [1], [0], [0], [1]])/np.sqrt(2)

baseline_qubits = 3

problem = QuantumLinearSystemProblem(problem_A_matrix, problem_b_vector)

Cannonical_HHL = HHL(get_result_function='get_fidelity_result',
                     eigenvalue_inversion=CannonicalInversion
                     )

fidelity = Cannonical_HHL.estimate(problem=problem, 
                                   num_clock_qubits=baseline_qubits,
                                   max_eigenvalue=1)
print(fidelity)

y_preprocessing=Lee_preprocessing(num_eval_qubits=baseline_qubits, backend=simulator, max_eigenvalue=1)

Yalovetsky_H_HHL = HHL(get_result_function='get_fidelity_result',
                       pre_processing=y_preprocessing.estimate,
                       eigenvalue_inversion=HybridInversion
                       )

hybrid_fidelity = Yalovetsky_H_HHL.estimate(problem=problem,
                                            num_clock_qubits=baseline_qubits,
                                            max_eigenvalue=1)
print(hybrid_fidelity)

e_preprocessing=Lee_preprocessing(num_eval_qubits=baseline_qubits+2, backend=simulator, max_eigenvalue=1)

Enhanced_H_HHL = HHL(get_result_function='get_fidelity_result',
                       pre_processing=y_preprocessing.estimate,
                       eigenvalue_inversion=EnhancedHybridInversion
                       )
enhanced_fidelity = Enhanced_H_HHL.estimate(problem=problem,
                                            num_clock_qubits=baseline_qubits,
                                            max_eigenvalue=1)
print(enhanced_fidelity)