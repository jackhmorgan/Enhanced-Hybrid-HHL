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

import __future__

from .quantum_linear_system import QuantumLinearSystemProblem, HHL_Result, QuantumLinearSystemSolver
 
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import partial_trace, Operator, Statevector
from qiskit.providers import Backend
from typing import Callable
from qiskit_ibm_runtime import Sampler
import numpy as np

from qiskit_aer import AerSimulator
simulator = AerSimulator()

def SwapTest(num_state_qubits : int)-> QuantumCircuit:
    """
    The function `SwapTest` creates a quantum circuit that performs a swap test between two sets of
    qubits.
    
    :param num_state_qubits: The `num_state_qubits` parameter in the `SwapTest` function represents the
    number of qubits used to encode the state you want to test. This function creates a quantum circuit
    that performs a swap test between the reference state encoded in the last qubit and the input state
    encoded in the first
    :return: The function `SwapTest(num_state_qubits)` returns a quantum circuit `st_circ` that performs
    a swap test between the state of the last qubit and the state of the first `num_state_qubits`
    qubits.
    """
    num_qubits = 2*num_state_qubits+1
    st_circ = QuantumCircuit(num_qubits)
    st_circ.h(-1)
    for i in range(num_state_qubits):
        st_circ.cswap(-1,i,num_state_qubits+i)
    st_circ.h(-1)
    return st_circ

def st_post_processing(result):
    if '0 1' in result.keys():
        counts_01 = result['0 1']
        if '1 1' in result.keys():
            counts_11 = result['1 1']
        else:
            counts_11 = 0

    else:
        counts_01 = result['1']
        counts_11 = result['3']
    if counts_01 <= counts_11:
        return 0
    else:
        prob_0 = counts_01/(counts_01+counts_11)
        return np.sqrt(2*prob_0 - 1)

def get_circuit_depth_result(backend: Backend) -> Callable:
    """
    The function `get_circuit_depth_result` takes a backend as input and returns a function that
    transpiles a given quantum circuit and calculates its depth.
    
    :param backend: The `backend` parameter in the `get_circuit_depth_result` function is expected to be
    an object of type `Backend`.
    :type backend: Backend
    :return: A callable function `get_result` is being returned, which takes a `QuantumCircuit` and a
    `QuantumLinearSystemProblem` as input and returns an `HHL_Result` object.
    """
    def get_result(hhl_circ : QuantumCircuit, 
                   problem : QuantumLinearSystemProblem,
                   ) -> HHL_Result:

        circuit = transpile(hhl_circ, backend)
        
        result = HHL_Result()
        result.circuit_results = circuit
        result.circuit_depth = circuit.depth()
        return result
    return get_result

def get_circuit_depth_result_st(backend : Backend, 
                                statevector : Statevector,
                                ) -> Callable:
    
    """
    The function `get_circuit_depth_result_st` takes a backend and statevector as input, and returns a
    function that calculates the depth of a quantum circuit with a SwapTest added to the end of it.
    
    :param backend: The `backend` parameter in the `get_circuit_depth_result_st` function refers to the
    target quantum device or simulator where the quantum circuit will be executed. It specifies the
    backend on which the quantum circuit will be transpiled and run
    :type backend: Backend
    :param statevector: The `statevector` parameter in the `get_circuit_depth_result_st` function
    represents the quantum state vector that you want to prepare in the quantum circuit. This state
    vector will be used as input to the quantum circuit that is being constructed within the
    `get_result` function
    :type statevector: Statevector
    :return: A callable function `get_result` is being returned, which takes `hhl_circ` and `problem` as
    input parameters and returns an `HHL_Result` object.
    """
    def get_result(hhl_circ, problem) -> HHL_Result:
        num_b_qubits = int(np.log2(len(problem.b_vector)))

        st = SwapTest(num_b_qubits)
        q_reg = QuantumRegister(st.num_qubits-num_b_qubits)
        c_reg = ClassicalRegister(1)

        hhl_circ.add_register(q_reg)
        hhl_circ.add_register(c_reg)

        hhl_circ.prepare_state(statevector, q_reg[:-1])
        hhl_circ.append(st, list(range(-st.num_qubits,0)))
        hhl_circ.measure(0,0)
        hhl_circ.measure(-1,c_reg[0])

        circuit = transpile(hhl_circ, backend)
        
        result = HHL_Result()
        result.circuit_results = circuit
        result.circuit_depth = circuit.depth()
        return result
    return get_result

def get_ionq_result_hhl(backend : Backend, 
                        statevector : Statevector,
                        ) -> Callable:
    """
    The function `get_ionq_result_hhl` prepares a quantum circuit for solving a problem using the HHL
    algorithm and returns the processed results evaluated on a qiskit_ionq backend with the SwapTest.
    
    :param backend: The `backend` parameter in the `get_ionq_result_hhl` function refers to the quantum
    backend on which the quantum circuit will be executed. This backend could be a simulator or a real
    quantum device provided by the IonQ platform or any other quantum computing service
    :type backend: Backend
    :param statevector: The `statevector` parameter in the `get_ionq_result_hhl` function is used to
    provide the quantum state vector that will be prepared in the quantum circuit before running the HHL
    algorithm. This state vector represents the input quantum state for the algorithm and is typically
    used to encode the problem
    :type statevector: Statevector
    :return: The `get_ionq_result_hhl` function returns the `get_result` function, which takes a HHL
    circuit and a problem as input and returns an `HHL_Result` object containing the circuit results and
    the number of results processed.
    """
    def get_result(hhl_circ, problem) -> HHL_Result:
        num_b_qubits = int(np.log2(len(problem.b_vector)))

        st = SwapTest(num_b_qubits)
        q_reg = QuantumRegister(st.num_qubits-num_b_qubits)
        c_reg = ClassicalRegister(1)

        hhl_circ.add_register(q_reg)
        hhl_circ.add_register(c_reg)

        hhl_circ.prepare_state(statevector, q_reg[:-1])
        hhl_circ.append(st, list(range(-st.num_qubits,0)))
        hhl_circ.measure(0,0)
        hhl_circ.measure(-1,c_reg[0])

        circuit = transpile(hhl_circ, backend=backend)
        
        job = backend.run(circuit)
        circuit_results = job.get_counts()
        circuit_depth = circuit.depth()

        result = HHL_Result()
        result.job_id = job.job_id()
        result.circuit_depth = circuit_depth
        result.circuit_results = circuit_results
        result.results_processed = st_post_processing(circuit_results)
        return result
    return get_result

def get_fidelity_result(hhl_circuit: QuantumCircuit, problem: QuantumLinearSystemProblem) -> HHL_Result:
    r'''Function to simulate the hhl_circuit, and return the inner product between the simulated estimate
    of |x> and the classically calculated solution.
    Args:
        '''

    # create projection operator of classically calculated solution
    ideal_x_operator = QuantumLinearSystemSolver(problem).ideal_x_statevector.to_operator()

    # qubits to remove in the partial trace of the simulated statevector. 
    trace_qubits = list(range(hhl_circuit.num_qubits-ideal_x_operator.num_qubits))

    # save statevector if the inversion was successful
    with hhl_circuit.if_test((0,1)) as passed:
        hhl_circuit.save_state()

    # transpile circuit
    circ = transpile(hhl_circuit, simulator)
    simulated_result = simulator.run(circ).result()

    # show the success probability of inversion
    circuit_results = simulated_result.get_counts()

    # retrieve statevector
    simulated_statevector = simulated_result.get_statevector()
    partial_simulated_density_matrix = partial_trace(simulated_statevector, trace_qubits)

    # square root of the expectation value is the fidelity
    result_processed = np.sqrt(partial_simulated_density_matrix.expectation_value(ideal_x_operator))

    result = HHL_Result()
    result.circuit_results = circuit_results
    result.results_processed = result_processed
    result.circuit_depth = circ.depth()
    
    return result

def get_simulator_result(observable: Operator) -> Callable:
    r'''Function to simulate the hhl_circuit, and return the inner product between the simulated estimate
    of |x> and the classically calculated solution.
    Args:
        '''
    def get_simulator_result_function(hhl_circuit: QuantumCircuit, problem: QuantumLinearSystemProblem) -> HHL_Result:
        # qubits to remove in the partial trace of the simulated statevector. 
        trace_qubits = list(range(hhl_circuit.num_qubits-observable.num_qubits))

        # save statevector if the inversion was successful
        with hhl_circuit.if_test((0,1)) as passed:
            hhl_circuit.save_state()

        # transpile circuit
        circ = transpile(hhl_circuit, simulator)
        simulated_result = simulator.run(circ).result()

        # show the success probability of inversion
        circuit_results = simulated_result.get_counts()

        # retrieve statevector 
        simulated_statevector = simulated_result.get_statevector()
        partial_simulated_density_matrix = partial_trace(simulated_statevector, trace_qubits)

        # square root of the expectation value is the fidelity
        result_processed = np.sqrt(partial_simulated_density_matrix.expectation_value(observable))

        result = HHL_Result()
        result.circuit_results = circuit_results
        result.results_processed = result_processed
        result.circuit_depth = circ.depth()
        
        return result
    return get_simulator_result_function

def get_swap_test_result(backend : Backend, 
                         statevector,
                         ) -> Callable:
    
    def get_result(hhl_circ, problem) -> HHL_Result:
        num_b_qubits = int(np.log2(len(problem.b_vector)))

        st = SwapTest(num_b_qubits)
        q_reg = QuantumRegister(st.num_qubits-num_b_qubits)
        c_reg = ClassicalRegister(1)

        hhl_circ.add_register(q_reg)
        hhl_circ.add_register(c_reg)

        with hhl_circ.if_test((0,1)) as passed:
            hhl_circ.prepare_state(statevector, q_reg[:-1])
            hhl_circ.append(st, range(-st.num_qubits,0))
            hhl_circ.measure(-1,c_reg[0])

        circuit = transpile(hhl_circ, backend)
        
        job = backend.run(circuit)
        circuit_results = job.result().get_counts()

        result = HHL_Result()
        result.circuit_results = circuit_results
        result.results_processed = st_post_processing(circuit_results)
        result.job_id = job.job_id()
        return result
    return get_result


def get_session_result(session, statevector) -> Callable:
    backend = session.service.get_backend(session.backend())
    sampler = Sampler(session=session)

    def get_result(hhl_circ, problem) -> HHL_Result:
        num_b_qubits = int(np.log2(len(problem.b_vector)))

        st = SwapTest(num_b_qubits)
        q_reg = QuantumRegister(st.num_qubits-num_b_qubits)
        c_reg = ClassicalRegister(1)

        hhl_circ.add_register(q_reg)
        hhl_circ.add_register(c_reg)

        hhl_circ.prepare_state(statevector, q_reg[:-1])
        hhl_circ.append(st, range(-st.num_qubits,0))
        hhl_circ.measure(-1,c_reg[0])

        circuit = transpile(hhl_circ, backend)
        
        job = sampler.run(circuit)

        result = HHL_Result()
        result.circuit_results = job.result().quasi_dists[0]
        result.job_id = job.job_id()
        result.circuit_depth = circuit.depth()
        return result
    return get_result

pass
