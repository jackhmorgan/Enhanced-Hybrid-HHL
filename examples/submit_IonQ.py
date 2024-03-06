import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')

from enhanced_hybrid_hhl import (HHL, 
                                 EnhancedHybridInversion,
                                 HybridInversion,
                                 CannonicalInversion,
                                 Lee_preprocessing,
                                 Yalovetsky_preprocessing,
                                 ideal_preprocessing,
                                 QuantumLinearSystemProblem,
                                 ExampleQLSP,
                                 QuantumLinearSystemSolver)
import numpy as np
from qiskit.quantum_info import random_hermitian
from qiskit.circuit.library import StatePreparation, HamiltonianGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ionq.ionq_provider import IonQProvider

ionq_provider = IonQProvider(token='eqMVRwVhZkVIX9lEBlIMYhxFtneQnCD5')

backend = ionq_provider.get_backend('ionq_qpu')

test_eigenvalues = [-1, -0.6, -0.4, -1/3, 1/3, 0.4, 0.6, 1]
problem_A_matrix = np.diag(test_eigenvalues)
problem_b_vector = np.array([[0], [1], [0], [0], [0], [1], [1], [0], [0], [0]])/np.sqrt(2)

test_eigenvalues = [-1, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -1/3, 1/3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1]
problem_A_matrix = np.diag(test_eigenvalues)
problem_b_vector = np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0]])/np.sqrt(2)

test_eigenvalues = [-1, -19/24, -0.75, -1/3, 1/3, 11/24, 0.5, 1]
problem_A_matrix = np.diag(test_eigenvalues)
problem_b_vector = np.array([[0], [1], [0], [0], [0], [1], [0], [0]])/np.sqrt(2)

test_eigenvalues = [-19/24, -18/24, 8/24, 11/24]
problem_A_matrix = np.diag(test_eigenvalues)
problem_b_vector = np.array([[0], [1], [1], [0]])/np.sqrt(2)



baseline_qubits = 3

problem = QuantumLinearSystemProblem(problem_A_matrix, 
                                     problem_b_vector
                                     )
solution= QuantumLinearSystemSolver(problem)
ideal_x = solution.ideal_x_statevector

baseline_qubits = 3

Cannonical_HHL = HHL(get_result_function='get_ionq_result',
                     eigenvalue_inversion=CannonicalInversion,
                     backend=backend,
                     statevector=ideal_x
                     )

result = Cannonical_HHL.estimate(problem=problem, 
                                   num_clock_qubits=baseline_qubits,
                                   max_eigenvalue=1,
                                   quantum_conditional_logic=False)

print(result)


y_preprocessing=Lee_preprocessing(num_eval_qubits=baseline_qubits, 
                                  backend=backend, 
                                  max_eigenvalue=1)

Yalovetsky_H_HHL = HHL(get_result_function='get_ionq_result',
                       pre_processing=y_preprocessing.estimate,
                       eigenvalue_inversion=HybridInversion,
                       backend=backend,
                       statevector=ideal_x
                       )

hybrid_result = Yalovetsky_H_HHL.estimate(problem=problem,
                                            num_clock_qubits=baseline_qubits,
                                            max_eigenvalue=1,
                                            quantum_conditional_logic=False)

print(hybrid_result)



e_preprocessing=Lee_preprocessing(num_eval_qubits=baseline_qubits+2, 
                                  backend=backend, 
                                  max_eigenvalue=1)

Enhanced_H_HHL = HHL(get_result_function='get_ionq_result',
                       pre_processing=e_preprocessing.estimate,
                       eigenvalue_inversion=EnhancedHybridInversion,
                       backend=backend,
                       statevector=ideal_x
                       )
enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                            num_clock_qubits=baseline_qubits,
                                            max_eigenvalue=1,
                                            quantum_conditional_logic=False,
                                            )

print(enhanced_result)