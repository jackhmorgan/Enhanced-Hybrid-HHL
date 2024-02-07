 '''
 Copyright 2023 Jack Morgan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from .quantum_linear_system import QuantumLinearSystemProblem
from qiskit.quantum_info import random_hermitian
import numpy as np

def RandomQLSP(number_qubits: int,
               maximum_condition_number: float,
               maximum_iterations_number: int = 100) :
    """
    The function RandomQLSP generates a random quantum linear system problem with a specified number of
    qubits. The function randomly draws Hermitian matrices until it finds a well conditioned system.
    
    :param number_qubits: The parameter "number_qubits" represents the number of qubits in the quantum
    linear system problem. It determines the size of the problem and the dimension of the matrix and
    vector involved.
    :type number_qubits: int
    :param maximum_condition_number: The maximum condition number is a threshold value that determines
    when to stop searching for a suitable problem. The condition number is a measure of how sensitive a
    problem is to changes in its input. In this case, it is the ratio of the largest eigenvalue to the
    smallest eigenvalue of the matrix A.
    :type maximum_condition_number: float
    :param maximum_iterations_number: The maximum number of random draws to search for a suitably
    conditioned matrix.
    :type maximum_condition_number: int
    :return: a QuantumLinearSystemProblem object, which is created using the A_matrix and b_vector
    variables.
    """
    condition_number = maximum_condition_number+1
    size = 2**number_qubits
    iterations = 0

    while condition_number > maximum_condition_number:
        
        # raise error if the number of iterations is exceeded.
        iterations+=1
        if iterations > maximum_iterations_number:
            raise Exception('Suitable random hermitian not found after maximum number of iterations')
        
        # randomly draw hermitian matrix.
        A_matrix = np.asmatrix(random_hermitian(size).data)
        b_vector = np.asarray(np.random.rand(size)).reshape((size,1))
        # check condition number
        A_eigen = np.linalg.eigvals(A_matrix)
        condition_number = abs(max(A_eigen,key=abs)/min(A_eigen,key=abs))

    # randomly generate normal b vector
    b_vector = np.asarray(np.random.rand(size)).reshape((size,1)) 
    b_vector = b_vector/np.linalg.norm(b_vector)
    
    return QuantumLinearSystemProblem(A_matrix, b_vector)

def ExampleQLSP(lam: float) -> QuantumLinearSystemProblem:
    """
    The function ExampleQLSP creates a QuantumLinearSystemProblem object with a given lambda value.
    
    :param lam: The parameter `lam` (short for lambda is a float value that is used to define the elements of the matrix
    `A_matrix`. The linear system is defined in equation 16 of [1]
    :type lam: float
    :return: a QuantumLinearSystemProblem object.
    References:
        [1]: Lee, Y., Joo, J., & Lee, S. (2019). 
        Hybrid quantum linear equation algorithm and its experimental test on ibm quantum experience. 
        Scientific reports, 9(1), 4778.
        `arxiv:1807.10651 <https://arxiv.org/abs/1807.10651>`_.
    """
    
    A_matrix = np.asmatrix([[0.5, 0.5-lam],[0.5-lam,0.5]])
    b_vector = [1,0]
    return QuantumLinearSystemProblem(A_matrix, b_vector)
        


