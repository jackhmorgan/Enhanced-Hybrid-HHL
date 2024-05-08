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
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np

from qiskit.circuit.library import PhaseEstimation, StatePreparation, HamiltonianGate
from qiskit.quantum_info import Statevector
from qiskit_algorithms import AlgorithmError

from qiskit.circuit.library import ExactReciprocal
from .quantum_linear_system import QuantumLinearSystemProblem as QLSP
from .get_result import (get_session_result, 
                         get_circuit_depth_result, 
                         get_circuit_depth_result_st, 
                         get_swap_test_result, 
                         get_fidelity_result, 
                         get_simulator_result, 
                         get_ionq_result_hhl)
from abc import ABC
from typing import Callable, Union

# The HHL class implements various algorithms for solving quantum linear systems.
# The `HHL` class implements the HHL algorithm for solving linear systems of equations using quantum
# computers, with support for different pre-processing algorithms and inversion circuits.
class HHL(ABC):
    r'''This class implements the HHL, Hybrid HHL, or Enhanced Hybrid HHL algorithms. The variant 
    is determined by the choice of pre-processing algorithm and inversion circuit. 
    '''
    def __init__(self,
                get_result_function: str = None,
                preprocessing: Callable = None,
                eigenvalue_inversion: Callable = None,
                **kwargs,
                ) -> None:
        """
        The `__init__` function initializes an object with three optional functions: `get_result`,
        `preprocessing`, and `eigenvalue_inversion`.
        
        :param get_result: String parameter to identify the function used to retrieve the hardware results.
                can be set either when initializing the class or estimating a problem.
        :type get_result_function: String
        :param preprocessing: A function that takes in the quantum linear system problem and performs any
        necessary pre-processing steps. This could include calculating the eigenvalues of the matrix and
        projecting the vector |b> onto their respective eigenvectors. The function should return the
        eigenvalues and the projection of |b> onto the eigenv
        :type preprocessing: Callable
        :param eigenvalue_inversion: The `eigenvalue_inversion` parameter is a function that takes in the
        eigenvalues and their respective eigenvectors and outputs the eigenvalue inversion sub-circuit. This
        sub-circuit is used in the HHL algorithm to perform the inversion of the eigenvalues
        :type eigenvalue_inversion: Callable
        """
        if not get_result_function==None:
            self._get_result = self._get_result_type(get_result_function, **kwargs) # import the result function from get_result
        self._preprocessing = preprocessing
        self._eigenvalue_inversion = eigenvalue_inversion

    @property
    def get_result(self) -> Union[Callable, None]:
        """Get get_result.

        Returns:
            The get_result function to evaluate the circuits.
        """
        return self._get_result
    
    @get_result.setter
    def get_result(self, get_result: Callable) -> None:
        """Set get_result.

        Args:
            get_result: A function to evaluate the circuits.
        """
        self._get_result = get_result
        
    @property
    def preprocessing(self) -> Union[Callable, None]:
        """Get preprocessing.

        Returns:
            The preprocessing function to determine eigenvalues and eigenbasis projection.
        """
        return self._preprocessing
    
    @preprocessing.setter
    def preprocessing(self, preprocessing: Callable) -> None:
        """Set preprocessing.

        Args:
            preprocessing: A function to determine eigenvalues and eigenbasis projection.
        """
        self._preprocessing = preprocessing
    
    @property
    def eigenvalue_inversion(self) -> Union[Callable, None]:
        """Get the eigenvalue_inversion circuit.

        Returns:
            The eigenvalue_inversion subcircuit.
        """
        return self._eigenvalue_inversion
    
    @eigenvalue_inversion.setter
    def eigenvalue_inversion(self, eigenvalue_inversion: Callable) -> None:
        """Set the eigenvalue_inversion circuit.

        Args:
            eigenvalue_inversion:  the eigenvalue_inversion subcircuit.
        """
        self._eigenvalue_inversion = eigenvalue_inversion

    def _get_result_type(self, get_result_name, **kwargs):
        ''' Import a callable from get_result.py that inputs the hhl circuit and outputs an hhl result.'''
        if get_result_name == 'get_circuit_depth_result':
            return get_circuit_depth_result(kwargs['backend'])

        if get_result_name == 'get_circuit_depth_result_st':
            return get_circuit_depth_result_st(kwargs['backend'], kwargs['statevector'])

        if get_result_name == 'get_swaptest_result':
            return get_swap_test_result(kwargs['backend'], kwargs['statevector'])
        
        if get_result_name == 'get_ionq_result':
            return get_ionq_result_hhl(kwargs['backend'], kwargs['statevector'])
        
        if get_result_name == 'get_simulator_result':
            if 'statevector' in kwargs.keys():
                if isinstance(kwargs['statevector'], Statevector):
                    operator = kwargs['statevector'].to_operator()
                else:
                    operator = Statevector(kwargs['statevector']).to_operator()
                return get_simulator_result(operator=operator)
            else:
                return get_simulator_result(kwargs['operator'])
        
        if get_result_name == 'get_fidelity_result':
            return get_fidelity_result
        
        if get_result_name == 'get_session_result':
            return get_session_result(kwargs['session'], kwargs['statevector'])

        else:
            if 'statevector' in kwargs:
                if 'backend' in kwargs:
                    return get_swap_test_result(kwargs['backend'], kwargs['statevector'])
                else: 
                    return get_simulator_result(kwargs['statevector'])
            else:
                if 'backend' in kwargs:
                    raise AlgorithmError('Backend provided without a statevector to compare result to')
                else: 
                    return get_simulator_result(kwargs['statevector'])

    
    def construct_circuit(self, 
                          num_clock_qubits, 
                          eigenvalue_inversion, 
                          state_preparation, 
                          hamiltonian_simulation, 
                          quantum_conditional_logic=True,
                          ) -> QuantumCircuit:
        """
        The function constructs a QuantumCircuit for the HHL algorithm using the specified subcircuits
        and optional quantum conditional logic.
        
        :param num_clock_qubits: The parameter `num_clock_qubits` represents the number of qubits used
        for the Quantum Phase Estimation (QPE) algorithm. QPE is used to estimate the eigenvalues of the
        input matrix in the HHL algorithm. The more qubits you allocate for QPE, the higher the
        precision
        :param eigenvalue_inversion: The `eigenvalue_inversion` parameter is a subcircuit that performs
        the eigenvalue inversion step in the HHL algorithm. This step is crucial for solving linear
        systems of equations using quantum computers. It involves inverting the eigenvalues of a matrix,
        which is typically done using techniques like phase
        :param state_preparation: The `state_preparation` parameter is the subcircuit that prepares the
        initial state |b> in the HHL algorithm. This subcircuit is responsible for encoding the input
        vector into the quantum state
        :param hamiltonian_simulation: The `hamiltonian_simulation` parameter is the subcircuit that
        performs the simulation of the Hamiltonian matrix. This subcircuit is responsible for encoding
        the Hamiltonian matrix into a quantum state and applying the corresponding unitary evolution. It
        is an essential part of the HHL algorithm, as it
        :param quantum_conditional_logic: The `quantum_conditional_logic` parameter is a boolean flag
        that determines whether the evaluation method within the `get_result` function supports quantum
        conditional logic. If `quantum_conditional_logic` is set to `True`, then steps 4 and 5 of the
        circuit construction will be included,, defaults to True (optional)
        :return: a QuantumCircuit object.
        """

        # define QPE subcircuit
        qpe = PhaseEstimation(num_clock_qubits, hamiltonian_simulation)
        
        # define registers
        flag = QuantumRegister(1)
        clock_reg = QuantumRegister(num_clock_qubits)
        b_reg = QuantumRegister(qpe.num_qubits - num_clock_qubits)
        c_reg = ClassicalRegister(1)

        # create circuit and add registers
        circ = QuantumCircuit(flag, c_reg, name='hhl_circ')
        circ.add_register(clock_reg)
        circ.add_register(b_reg)

        # append steps 1-3
        circ.append(state_preparation, b_reg)
        circ.append(qpe, clock_reg[:]+b_reg[:])
        circ.append(eigenvalue_inversion, clock_reg[::-1]+[flag[0]]) # The ordering of the qubits is consistent with ExactReciprocal
       
        # steps 4 & 5 if conditional logic is supported       
        if quantum_conditional_logic:       
            circ.measure(0,0)
            with circ.if_test((0,1)) as passed:
                circ.append(qpe.inverse(), clock_reg[:]+b_reg[:])
        
        # step 5 if logic is not supported
        else:
            circ.append(qpe.inverse(), clock_reg[:]+b_reg[:])
            circ.measure(0,0)
        return circ
        
        
     
    def estimate(self, 
                 problem: QLSP,
                 num_clock_qubits: Union[int, None] = 3,
                 max_eigenvalue: Union[float, None] = None,
                 quantum_conditional_logic: bool = True,
                 get_result_function: str = None,
                 **kwargs
                 ):
        r''' 
        Run HHL algorithm with specified preprocessing and circuit construction.

        Args:
            problem: The Quantum Linear System Problem to be solved.
            num_clock_qubits: The number of qubits to evaluate eigenvalue inversion.
            max_eigenvalue: An upper bound to the eigenvalue of the matrix, used to calculate
                the hamiltonian simulation gate if none is provided.
            quantum_conditional_logic: Boolean flag of whether the evaluation method
                within get_result supports quantum conditional logic.
            get_result_function: String parameter to identify the function used to retrieve the hardware results.
                can be set either when initializing the class or estimating a problem.
        Returns:
            HHL_result
        '''

        if not get_result_function == None:
            self._get_result = self._get_result_type(get_result_function, **kwargs) # import the result function from get_result

        if self._get_result == None:
            raise AlgorithmError('get_result_function needs to be set either when initializing the class or estimating a problem')

        eigenvalue_list = None
        eigenbasis_projection_list = None
        
        # If Inversion circuit is set, perform preprocessing and construct inversion_circuit
        if getattr(self, 'eigenvalue_inversion', None) is not None:
            if getattr(self, 'preprocessing', None) is not None:
                eigenvalue_list, eigenbasis_projection_list = self.preprocessing(problem)
                if max_eigenvalue == None:
                    max_eigenvalue = max(eigenvalue_list, key = abs)
                elif not max_eigenvalue in eigenvalue_list:
                    eigenvalue_list.append(max_eigenvalue)
                    eigenbasis_projection_list.append(0)
                inversion_circuit = self.eigenvalue_inversion(eigenvalue_list, eigenbasis_projection_list, num_clock_qubits, **kwargs)

            else:
                inversion_circuit = self.eigenvalue_inversion(num_clock_qubits)
        # If no inversion circuit, use Gray Code
        else:
            inversion_circuit = ExactReciprocal(num_state_qubits=num_clock_qubits, scaling=2*2**-num_clock_qubits, neg_vals=True)    

        # If the hamiltonian simulation is not specified in the problem, use the standard HamiltonianGate
        if getattr(problem, 'hamiltonian_simulation', None) is None:
            if max_eigenvalue == None:
                raise AlgorithmError('An upper bound on the eigenvalues is needed')
            
            scale = abs((0.5-2**-num_clock_qubits)/max_eigenvalue)
            hamiltonian_simulation = HamiltonianGate(problem.A_matrix, -2*np.pi*scale)
        
        else:
            hamiltonian_simulation = problem.hamiltonian_simulation
        

        # If the state_preparation is not specified in the problem, use the standard StatePreparation
        
        if getattr(problem, 'state_preparation', None) is None:
            
            state_preparation = StatePreparation(Statevector(problem.b_vector))
        
        else:
            state_preparation = problem.state_preparation
    
        # construct circuit
        circuit = self.construct_circuit(num_clock_qubits, 
                                         inversion_circuit, 
                                         state_preparation, 
                                         hamiltonian_simulation, 
                                         quantum_conditional_logic,
                                         )
        # get result
        result = self.get_result(circuit, problem)

        # Add preprocessing data to result
        result.eigenvalue_list = eigenvalue_list
        result.eigenbasis_projection_list = eigenbasis_projection_list
        
        return result
pass
