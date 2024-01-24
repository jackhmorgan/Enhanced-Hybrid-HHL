import __future__

import sys
import os

# Assuming the parent directory of the tests folder is in your project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np

from qiskit.circuit.library import PhaseEstimation, StatePreparation
from qiskit.extensions import HamiltonianGate
from qiskit.quantum_info import Statevector
from qiskit_algorithms import AlgorithmError

from qiskit.circuit.library import ExactReciprocal
from QuantumLinearSystemProblem import QuantumLinearSystemProblem as QLSP
from abc import ABC
from typing import Callable, Union

class HHL(ABC):
    r'''This class implements the cannonical, hybrid, or enhanced hybrid variants of the HHL algorithm. The variant is determined by the choice of pre-processing and inversion
    circuit. By default it implements the complete Enhanced Hybrid algorithm with Quantum Condition Logic Quantum Phase Estimation.'''
    def __init__(self,
                get_result: Callable,
                pre_processing: Callable = None,
                eigenvalue_inversion: Callable = None,
                ):
        self._get_result = get_result
        self._pre_processing = pre_processing
        self._eigenvalue_inversion = eigenvalue_inversion

    @property
    def get_result(self) -> Union[Callable, None]:
        return self._get_result
    
    @get_result.setter
    def get_result(self, get_result: Callable) -> None:
        self._get_result = get_result
        
    @property
    def pre_processing(self) -> Union[Callable, None]:
        return self._pre_processing
    
    @pre_processing.setter
    def pre_processing(self, pre_processing: Callable) -> None:
        self._pre_processing = pre_processing
    
    @property
    def eigenvalue_inversion(self) -> Union[Callable, None]:
        return self._eigenvalue_inversion
    
    @eigenvalue_inversion.setter
    def eigenvalue_inversion(self, eigenvalue_inversion: Callable) -> None:
        self._eigenvalue_inversion = eigenvalue_inversion

    def construct_circuit(self, num_clock_qubits, eigenvalue_inversion, state_preparation, hamiltonian_simulation, quantum_conditional_logic=True) -> QuantumCircuit:
        '''This method creates the HHL circuit'''
        qpe = PhaseEstimation(num_clock_qubits, hamiltonian_simulation)
        
        flag = QuantumRegister(1)
        clock_reg = QuantumRegister(num_clock_qubits)
        b_reg = QuantumRegister(qpe.num_qubits - num_clock_qubits)
        c_reg = ClassicalRegister(1)

        circ = QuantumCircuit(flag, c_reg, name='hhl_circ')
        circ.add_register(clock_reg)
        circ.add_register(b_reg)
        circ.append(state_preparation, b_reg)
        circ.append(qpe, clock_reg[:]+b_reg[:])
        circ.append(eigenvalue_inversion, clock_reg[::-1]+[flag[0]])
       
        if quantum_conditional_logic:       
            circ.measure(0,0)
            with circ.if_test((0,1)) as passed:
                circ.append(qpe.inverse(), clock_reg[:]+b_reg[:])
        else:
            circ.append(qpe.inverse(), clock_reg[:]+b_reg[:])
        return circ
        
        
     
    def estimate(self, 
                 problem: QLSP,
                 num_clock_qubits: Union[int, None] = 3,
                 max_eigenvalue: Union[float, None] = None,
                 quantum_conditional_logic: bool = True,
                 ):
        r''' Runs QPE_QCL and uses the results to run NISQ_HHL'''

        eigenvalue_list = None
        eigenbasis_projection_list = None
        
        if getattr(self, 'eigenvalue_inversion', None) is not None:
            if getattr(self, 'pre_processing', None) is not None:
                eigenvalue_list, eigenbasis_projection_list = self.pre_processing(problem)

            max_eigenvalue = max(eigenvalue_list, key = abs)
            inversion_circuit = self.eigenvalue_inversion(eigenvalue_list, eigenbasis_projection_list, num_clock_qubits)

        else:
            inversion_circuit = ExactReciprocal(num_state_qubits=num_clock_qubits, scaling=2*2**-num_clock_qubits, neg_vals=True)    

        if getattr(problem, 'hamiltonian_simulation', None) is None:
            if max_eigenvalue == None:
                raise AlgorithmError('An upper bound on the eigenvalues is needed')
            
            scale = abs((0.5-2**-num_clock_qubits)/max_eigenvalue)
            hamiltonian_simulation = HamiltonianGate(problem.A_matrix, -2*np.pi*scale)
        
        else:
            hamiltonian_simulation = problem.hamiltonian_simulation
        
        if getattr(problem, 'state_preparation', None) is None:
            
            state_preparation = StatePreparation(Statevector(problem.b_vector))
        
        else:
            state_preparation = problem.state_preparation
    
        #print('scale=',self.scale)
        circuit = self.construct_circuit(num_clock_qubits, 
                                         inversion_circuit, 
                                         state_preparation, 
                                         hamiltonian_simulation, 
                                         quantum_conditional_logic,
                                         )
        result = self.get_result(circuit, problem)

        result.eigenvalue_list = eigenvalue_list
        result.eigenbasis_projection_list = eigenbasis_projection_list
        
        return result
pass