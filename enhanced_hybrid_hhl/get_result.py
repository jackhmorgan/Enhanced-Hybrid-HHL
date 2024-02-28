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
from qiskit.quantum_info import partial_trace, Operator
from typing import Callable
import numpy as np

from qiskit import Aer
simulator = Aer.get_backend('aer_simulator')


from qiskit_ionq.ionq_provider import IonQProvider
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
ionq_provider = IonQProvider(token='W8fkNnlIqaWDP83QCPTnP5HjoELVXMbP')
ionq_simulator = ionq_provider.get_backend('ionq_simulator')

def get_circuit_depth_result(backend):
  
    def get_result(hhl_circ, problem) -> HHL_Result:
        num_b_qubits = int(np.log2(len(problem.b_vector)))

        circuit = transpile(hhl_circ, backend)
        
        result = HHL_Result()
        result.circuit_results = circuit
        result.results_processed = circuit.depth()
        return result
    return get_result

def get_circuit_depth_result_st(backend, statevector):
    def SwapTest(num_state_qubits):
        num_qubits = 2*num_state_qubits+1
        st_circ = QuantumCircuit(num_qubits)
        st_circ.h(-1)
        for i in range(num_state_qubits):
            st_circ.cswap(-1,i,num_state_qubits+i)
        st_circ.h(-1)
        return st_circ
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
        result.results_processed = circuit.depth()
        return result
    return get_result

def get_ionq_result_hhl(backend, statevector):
    def SwapTest(num_state_qubits):
        num_qubits = 2*num_state_qubits+1
        st_circ = QuantumCircuit(num_qubits)
        st_circ.h(-1)
        for i in range(num_state_qubits):
            st_circ.cswap(-1,i,num_state_qubits+i)
        st_circ.h(-1)
        return st_circ

    def st_post_processing(result = None, counts_01=None, counts_11=None):
        if not result==None:
            if '0 1' in result.keys():
                counts_01 = result['0 1']
            else:
                counts_01 = 0
                
            if '1 1' in result.keys():
                counts_11 = result['1 1']
            else:
                if counts_01 == 0:
                    counts_11 = 1
                else:
                    counts_11 = 0
        
        prob_0 = counts_01/(counts_01+counts_11)
        if prob_0 == 0:
            return 0
        return np.sqrt(2*prob_0 - 1)

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

        circuit = transpile(hhl_circ, ionq_simulator)
        
        circuit_result = ionq_simulator.run(circuit).result().get_counts()
        result_processed = st_post_processing(result = circuit_result)

        result = HHL_Result()
        result.circuit_results = circuit_result
        result.results_processed = result_processed
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
        
        return result
    return get_simulator_result_function

def get_swap_test_result(backend, statevector) -> Callable:

    def SwapTest(num_state_qubits):
        num_qubits = 2*num_state_qubits+1
        st_circ = QuantumCircuit(num_qubits)
        st_circ.h(-1)
        for i in range(num_state_qubits):
            st_circ.cswap(-1,i,num_state_qubits+i)
        st_circ.h(-1)

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

        if getattr(problem, 'post_processing', None) is not None:
            processed_result = problem.post_processing(processed_result)
        
        result = HHL_Result()

        result.circuit_results = circuit_result
        result.results_processed = processed_result

        return result
    
    return get_result
pass
