import __future__

import sys
import os

# Assuming the parent directory of the tests folder is in your project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from QuantumLinearSystemProblem import QuantumLinearSystemProblem
from QuantumLinearSystemSolver import QuantumLinearSystemSolver
from QuantumLinearSystemResult import HHL_Result
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import partial_trace
from typing import Callable
from SwapTest import SwapTest
import numpy as np

from qiskit import Aer
simulator = Aer.get_backend('aer_simulator')

def get_fidelity_result(hhl_circuit: QuantumCircuit, problem: QuantumLinearSystemProblem) -> HHL_Result:

    ideal_x_operator = QuantumLinearSystemSolver(problem).ideal_x_statevector.to_operator()
    trace_qubits = list(range(1+hhl_circuit.num_qubits-ideal_x_operator.dim[0]))
    with hhl_circuit.if_test((0,1)) as passed:
        hhl_circuit.save_state()

    circ = transpile(hhl_circuit, simulator)
    simulated_result = simulator.run(circ).result()

    circuit_results = simulated_result.get_counts()

    simulated_statevector = simulated_result.get_statevector()
    partial_simulated_density_matrix = partial_trace(simulated_statevector, trace_qubits)
    result_processed = np.sqrt(partial_simulated_density_matrix.expectation_value(ideal_x_operator))

    result = HHL_Result()
    result.circuit_results = circuit_results
    result.results_processed = result_processed
    
    return result

def get_swap_test_result(backend, statevector) -> Callable:

    def st_post_processing(result = None, counts_01=None, counts_11=None):
        if not result==None:
            counts_01 = result['01']
            counts_11 = result['11']
        prob_0 = counts_01/(counts_01+counts_11)
        return np.sqrt(2*prob_0 - 1)

    def get_result(hhl_circ, problem) -> HHL_Result:
        num_b_qubits = len(problem.b_vector)

        st = SwapTest(num_b_qubits)
        q_reg = QuantumRegister(st.num_qubits-num_b_qubits)
        c_reg = ClassicalRegister(1)

        hhl_circ.add_register(q_reg)
        hhl_circ.add_register(c_reg)

        with hhl_circ.if_test((0,1)) as passed:
            hhl_circ.prepare_state(statevector, q_reg[:-1])
            hhl_circ.append(st, range(-st.num_qubits))
            hhl_circ.measure(-1,c_reg[0])

        circuit = transpile(hhl_circ, backend)
        
        circuit_result = backend.run(circuit).result()
        result_processed = st_post_processing(circuit_result.get_counts())

        result = HHL_Result()
        result.circuit_results = circuit_result
        result.results_processed = result_processed
        return result
    return get_result

def get_estimator_result(backend, observable):

    from qiskit_ibm_runtime import Estimator
    from qiskit.quantum_info import Pauli
    
    estimator = Estimator(backend=backend)
    
    def get_result(hhl_circ, problem):

        pauli_list = ['I' for _ in range(hhl_circ.num_qubits)]
        pauli_list[-1] = 'Z'
        norm_observable = Pauli("".join(pauli_list))

        circuit = transpile(hhl_circ, backend)
        job = estimator.run([circuit, circuit], [observable, norm_observable])
        circuit_result = job.result().values[0]
        norm_result = job.result().values[1]

        processed_result = norm_result

        if getattr(problem, 'pose_processing', None) is not None:
            processed_result = problem.post_processing(processed_result)
        
        result = HHL_Result()

        result.circuit_results = circuit_result
        result.results_processed = processed_result

        return result
    
    return get_result
pass