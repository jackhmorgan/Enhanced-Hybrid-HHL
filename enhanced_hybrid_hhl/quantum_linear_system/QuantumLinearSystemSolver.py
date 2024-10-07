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
from .quantum_linear_system import QuantumLinearSystemProblem, HHL_Result
import numpy as np
from qiskit.quantum_info import Statevector

def QuantumLinearSystemSolver(problem: QuantumLinearSystemProblem) -> HHL_Result:
    
    A_matrix = np.array(problem.A_matrix)
    if not isinstance(A_matrix, np.ndarray):
        raise ValueError('QuantumLinearSytemSolver requires an explicit A_matrix')

    b_vector = np.array(problem.b_vector)
    if not isinstance(b_vector, np.ndarray):
        raise ValueError('QuantumLinearSytemSolver requires an explicit b_vector')
    
    if not b_vector.ndim == 1:
        if b_vector.ndim == 2:
            if not b_vector.shape[1] == 1:
                raise ValueError('b_vector must be a vector')
        else:
            raise ValueError('b_vector must be a vector')
    
    else:
        b_vector.reshape(-1,1)

    A_eigen = np.linalg.eigh(A_matrix)

    b_norm = np.linalg.norm(b_vector) 
    b_state = b_vector/b_norm
    b_eigen = np.linalg.inv(A_eigen[1]).dot(b_state)
    
    eigenvalue_list, eigenbasis_projection_list = [], []

    for i, amp in enumerate(b_eigen):
        eigenvalue_list.append(A_eigen[0][i])
        eigenbasis_projection_list.append(abs(amp))
    

    x = np.linalg.solve(A_matrix, b_state)
    x_norm = np.linalg.norm(x)
    x_state = x/x_norm
    x_meas = []
    for amp in x_state[:]:
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