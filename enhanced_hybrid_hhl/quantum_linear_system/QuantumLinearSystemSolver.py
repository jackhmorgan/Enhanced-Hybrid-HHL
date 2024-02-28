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
from quantum_linear_system import QuantumLinearSystemProblem, HHL_Result
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

if __name__ == 'main':
    A = np.array([[1,0.33],[0.33,1]])
    b = [1,0]
    example_problem = QuantumLinearSystemProblem(A_matrix = a,
                                                 b_vector = b)
    solution = QuantumLinearSystemSolver(example_problem)
