from .quantum_linear_system import QuantumLinearSystemProblem
from qiskit.quantum_info import random_hermitian
import numpy as np

def RandomQLSP(number_qubits,
               maximum_condition_number) :
    condition_number = maximum_condition_number+1
    size = 2**number_qubits
    iterations = 0
    while condition_number > maximum_condition_number:
        if iterations%10 == 0 and not iterations == 0:
            print('Still searching, iterations: ', iterations)
        iterations+=1

        A_matrix = np.asmatrix(random_hermitian(size).data)
        b_vector = np.asarray(np.random.rand(size)).reshape((size,1))

        A_eigen = np.linalg.eigvals(A_matrix)
        condition_number = abs(max(A_eigen,key=abs)/min(A_eigen,key=abs))

    b_vector = np.asarray(np.random.rand(size)).reshape((size,1)) 
    b_vector = b_vector/np.linalg.norm(b_vector)
    
    return QuantumLinearSystemProblem(A_matrix, b_vector)

def ExampleQLSP(lam) -> QuantumLinearSystemProblem:
    A_matrix = np.asaray([[0.5, 0.5-lam],[0.5-lam,0.5]])
    b_vector = [1,0]
    return QuantumLinearSystemProblem(A_matrix, b_vector)
        


