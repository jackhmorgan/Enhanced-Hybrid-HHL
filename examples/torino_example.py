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
from qiskit_ibm_runtime import Session

from qiskit_ibm_runtime import QiskitRuntimeService, Session

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-ncsu/nc-state/amplitude-estima',
    token='35e7316e04600dcc5e1b36a25b8eb9c61601d1456fe4d0ddb1604a8ef1847f0614516de700c3f5d9bac07141ade760115a07f1a706307d04fd16066d1a0dd109'
)

backend = service.get_backend('ibm_torino')
#backend = service.get_backend('ibmq_qasm_simulator')

test_eigenvalues = [-21/24, -20/24, -19/24, -18/24, 8/24, 10/24, 11/24, 12/24]
A_matrix_3 = np.diag(test_eigenvalues)
b_vector = np.array([[0], [1], [0], [0], [0], [1], [0], [0]])/np.sqrt(2)

test_eigenvalues = [-19/24, -18/24, 8/24, 11/24]
A_matrix_2 = np.diag(test_eigenvalues)
b_vector = np.array([[0], [1], [1], [0]])/np.sqrt(2)

problem = QuantumLinearSystemProblem(A_matrix_2, 
                                     b_vector
                                     )

solution= QuantumLinearSystemSolver(problem)
ideal_x = solution.ideal_x_statevector

baseline_qubits = 3

with Session(service=service, backend=backend) as session:

    Cannonical_HHL = HHL(get_result_function='get_session_result',
                        eigenvalue_inversion=CannonicalInversion,
                        session=session,
                        statevector = ideal_x)

    result = Cannonical_HHL.estimate(problem=problem, 
                                    num_clock_qubits=baseline_qubits,
                                    max_eigenvalue=1,
                                    quantum_conditional_logic=False,
                                    )

    print(result)


    y_preprocessing=Lee_preprocessing(num_eval_qubits=baseline_qubits,  
                                      max_eigenvalue=1,
                                      session=session,
                                      )

    Yalovetsky_H_HHL = HHL(get_result_function='get_session_result',
                        pre_processing=y_preprocessing.estimate,
                        eigenvalue_inversion=HybridInversion,
                        session=session,
                        statevector = ideal_x)

    hybrid_result = Yalovetsky_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=baseline_qubits,
                                                max_eigenvalue=1,
                                                quantum_conditional_logic=False,
                                                )

    print(hybrid_result)


    e_preprocessing=Lee_preprocessing(num_eval_qubits=baseline_qubits+2,  
                                      max_eigenvalue=1,
                                      session=session, 
                                      )

    Enhanced_H_HHL = HHL(get_result_function='get_session_result',
                        pre_processing=e_preprocessing.estimate,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        session=session,
                        statevector = ideal_x)
    enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=baseline_qubits,
                                                max_eigenvalue=1,
                                                quantum_conditional_logic=False,
                                                )

    print(enhanced_result)