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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

def EnhancedHybridInversion(eigenvalue_list, eigenbasis_projection_list, num_clock_qubits, **kwargs) -> QuantumCircuit:
    """
    The function `HybridInversion` constructs a quantum circuit for hybrid inversion based on given
    eigenvalues, eigenbasis projections, and number of clock qubits.
    
    :param eigenvalue_list: The `eigenvalue_list` parameter contains a list of eigenvalues
    relevant to the linear system.
    :param eigenbasis_projection_list: The `eigenbasis_projection_list` contains the
    projections of the b vector onto their respective eigenvectors
    :param num_clock_qubits: The `num_clock_qubits` parameter represents the number of clock qubits in
    the quantum circuit. 
    :return: The function `HybridInversion` is returning a `QuantumCircuit` object that represents a
    quantum circuit implementing a hybrid inversion operation. If the precision of the eigenvalue list is greater than
    num_clock_qubits, then the enhancement is automatically used to calculate the inversion angles.
    """
    probability_threshold = 0
    if 'probability_threshold' in kwargs.keys():
        probability_threshold = kwargs['probability_threshold']
    
    control_state_list, rotation_angle_list = enhanced_angle_processing_practical(eigenvalue_list, 
                                                                                  eigenbasis_projection_list, 
                                                                                  num_clock_qubits,
                                                                                  probability_threshold=probability_threshold)
    circ = QuantumCircuit(num_clock_qubits+1, name='hybrid_inversion')

    for state, angle in zip(control_state_list, rotation_angle_list):

        gate = RYGate(angle).control(num_ctrl_qubits=num_clock_qubits, label='egn_inv', ctrl_state=state)
        circ.append(gate, circ.qubits)

    return circ

def HybridInversion(eigenvalue_list, eigenbasis_projection_list, num_clock_qubits) -> QuantumCircuit:
    """
    The function `HybridInversion` constructs a quantum circuit for hybrid inversion based on given
    eigenvalues, eigenbasis projections, and number of clock qubits.
    
    :param eigenvalue_list: The `eigenvalue_list` parameter contains a list of eigenvalues
    relevant to the linear system.
    :param eigenbasis_projection_list: The `eigenbasis_projection_list` contains the
    projections of the b vector onto their respective eigenvectors
    :param num_clock_qubits: The `num_clock_qubits` parameter represents the number of clock qubits in
    the quantum circuit. 
    :return: The function `HybridInversion` is returning a `QuantumCircuit` object that represents a
    quantum circuit implementing a hybrid inversion operation. If the precision of the eigenvalue list is greater than
    num_clock_qubits, then the enhancement is automatically used to calculate the inversion angles.
    """

    scale = abs((0.5-2**-num_clock_qubits)/abs(max(eigenvalue_list, key=abs)))

    eigenvalues = [eigen for i, eigen in enumerate(eigenvalue_list) if 0 < abs(eigen) < eigenbasis_projection_list[i]*(2**num_clock_qubits)]
    control_state_list = [round(value*scale*(2**(num_clock_qubits)-1)) for value in eigenvalues]
    circ = QuantumCircuit(num_clock_qubits+1, name='hybrid_inversion')

    for state in control_state_list:
        if state == 0:
            continue
        angle = 2*np.arcsin(1/state)

        if state <= 0:
            state = round(state + (2**num_clock_qubits))
        gate = RYGate(angle).control(num_ctrl_qubits=num_clock_qubits, label='egn_inv', ctrl_state=state)
        circ.append(gate, circ.qubits)

    return circ


def CannonicalInversion(num_clock_qubits) -> QuantumCircuit:
    """
    The function `CannonicalInversion` creates a quantum circuit for performing canonical inversion on a
    specified number of clock qubits.
    
    :param num_clock_qubits: The `num_clock_qubits` parameter represents the number of clock qubits in
    the quantum circuit. This parameter is used to determine the size of the quantum circuit and the
    operations to be performed on it
    :return: A QuantumCircuit named 'hybrid_inversion' with controlled RY gates applied for each state
    in the specified number of clock qubits.
    """
    
    circ = QuantumCircuit(num_clock_qubits+1, name='hybrid_inversion')

    for state in range(2**num_clock_qubits):
        value = state
        if value == 0:
            continue
        if value >= (2**(num_clock_qubits-1)):
            value = state - (2**(num_clock_qubits))
        angle = 2*np.arcsin(1/value)
        gate = RYGate(angle).control(num_ctrl_qubits=num_clock_qubits, label='egn_inv', ctrl_state=state)
        circ.append(gate, circ.qubits)

    return circ

def GrayCodeInversion(num_clock_qubits):
    """
    The function `GrayCodeInversion` returns an ExactReciprocal object with specified parameters.
    
    :param num_clock_qubits: The `num_clock_qubits` parameter represents the number of qubits used in
    the quantum circuit. In this context, it is used to create an `ExactReciprocal` circuit from the
    Qiskit library. This circuit is designed to perform the reciprocal operation with a specified number
    of qubits
    :return: The function `GrayCodeInversion` is returning an instance of the `ExactReciprocal` class
    from the Qiskit library.
    """
    from qiskit.circuit.library import ExactReciprocal
    er = ExactReciprocal(num_clock_qubits, 2*2**-num_clock_qubits, neg_vals=True)
    return er

def enhanced_angle_processing_practical(eigenvalue_list,
                                        eigenbasis_projection_list, 
                                        num_clock_qubits, 
                                        probability_threshold=0) -> tuple((list[int], list[float])):
    """
    The function `enhanced_angle_processing_practical` calculates control state and rotation angles
    based on eigenvalues and eigenbasis projections.
    
    :param eigenvalue_list: Eigenvalues of a quantum state represented as a list of complex numbers
    :param eigenbasis_projection_list: The `eigenbasis_projection_list` parameter in the
    `enhanced_angle_processing_practical` function represents the list of projections of the eigenbasis
    onto the quantum state. Each element in this list corresponds to the projection of the corresponding
    eigenvalue onto the quantum state
    :param num_clock_qubits: The `num_clock_qubits` parameter represents the number of qubits used for
    to represent the eigenvalues. This parameter is used in the
    `enhanced_angle_processing_practical` function to calculate certain scaling factors and perform
    quantum operations based on the number of clock qubits provided
    :param probability_threshold: The `probability_threshold` parameter in the
    `enhanced_angle_processing_practical` function is a threshold value that determines whether a
    particular state and its corresponding rotation angle should be included in the final output. If the
    product of the sum of probabilities for a state and its final amplitude is greater than the `,
    defaults to 0 (optional)
    :return: The function `enhanced_angle_processing_practical` returns a tuple containing two lists:
    `control_state_list` and `rotation_angle_list`. The `control_state_list` contains the control states
    for the quantum circuit, and the `rotation_angle_list` contains the rotation angles corresponding to
    each control state.
    """
    
    clock = num_clock_qubits
    scale = abs((0.5-2**-clock)/abs(max(eigenvalue_list, key=abs)))
    
    constant = 1
    
    probability_dictionary = {}
    final_amplitude_dictionary = {}

    for value, projection in zip(eigenvalue_list, eigenbasis_projection_list):

        if projection==0:
            continue

        state = value*scale*(2**(clock)-1)

        state_floor = int(np.floor(state))
        state_ciel = int(np.ceil(state))

        increment = abs(state_ciel-state_floor)
        
        for ctrl_state in [state_floor, state_ciel]:
            if not ctrl_state == 0:
                if increment==0:
                    weight = 1/2
                else:
                    weight = 1-(abs(state - ctrl_state)/increment)

                if ctrl_state < 0:
                    ctrl_state += int(2**clock)

                if ctrl_state not in probability_dictionary.keys():
                    probability_dictionary[ctrl_state] = {}

                probability_dictionary[ctrl_state][value] = projection*weight
        
    amplitude_dictionary = {}
        
    for state, vectors in probability_dictionary.items():
        
        ave_eigenvalue = np.average(list(vectors.keys()), weights = list(vectors.values()))
        final_amplitude = constant / ave_eigenvalue
        if abs(sum(vectors.values())*final_amplitude) > probability_threshold:
            amplitude_dictionary[state] = final_amplitude

    control_state_list, rotation_angle_list = [], []

    constant = abs(1/max(amplitude_dictionary.values(), key=abs))
    for state, amplitude in amplitude_dictionary.items():
        control_state_list.append(state)
        rotation_angle_list.append(2*np.arcsin(constant*amplitude))
    
    return control_state_list, rotation_angle_list
