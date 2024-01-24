
import sys
import os

# Assuming the parent directory of the tests folder is in your project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from enhanced_hybrid_hhl.QuantumLinearSystemProblem import QuantumLinearSystemProblem
from enhanced_hybrid_hhl.QuantumLinearSystemResult import HHL_Result
from qiskit_algorithms.exceptions import AlgorithmError
import numpy as np
from qiskit.quantum_info import Statevector

def QuantumLinearSystemSolver(problem: QuantumLinearSystemProblem) -> HHL_Result:
    
    A_matrix = problem.A_matrix
    if not type(A_matrix) == np.matrix:
        raise AlgorithmError('QuantumLinearSytemSolver requires an explicit A_matrix')

    b_vector = problem.b_vector
    if not type(b_vector) == np.matrix:
        raise AlgorithmError('QuantumLinearSytemSolver requires an explicit b_vector')

    A_eigen = np.linalg.eigh(A_matrix)

    b_norm = np.linalg.norm(b_vector) 
    b_state = b_vector/b_norm
    b_eigen = np.linalg.inv(A_eigen[1]).dot(b_state)
    
    eigenvalue_list, eigenbasis_projection_list = [], []

    for i, amp in enumerate(b_eigen):
        eigenvalue_list.append(A_eigen[0][i])
        eigenbasis_projection_list.append(abs(amp)[0,0])
    

    x = np.linalg.solve(A_matrix, b_state)
    x_norm = np.linalg.norm(x)
    x_state = x/x_norm
    x_meas = []
    for amp in x_state[:,0]:
        x_meas.append(abs(amp*np.conj(amp)))
    ideal_x_statevector = Statevector(x_state)
    circuit_results = x_meas

    result = HHL_Result()
    result.circuit_results = circuit_results
    result.eigenvalue_list = eigenvalue_list
    result.eigenbasis_projection_list = eigenbasis_projection_list
    result.ideal_x_statevector = ideal_x_statevector
    return result
pass