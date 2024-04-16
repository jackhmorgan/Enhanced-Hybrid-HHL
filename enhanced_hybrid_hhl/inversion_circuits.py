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

def EnhancedHybridInversion(eigenvalue_list : list, 
                            eigenbasis_projection_list: list, 
                            num_clock_qubits : int, 
                            **kwargs) -> QuantumCircuit:
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
    # Set parameters for enhancement calculation if they are provided
    probability_threshold = 0 # relavence threshold for the second layer of filtering
    exact_alpha = False # Boolean variable. If true then use the exact function for alpha.
    if 'probability_threshold' in kwargs.keys():
        probability_threshold = kwargs['probability_threshold']
    if 'exact_alpha' in kwargs.keys():
        exact_alpha = kwargs['exact_alpha']
    
    control_state_list, rotation_angle_list = Enhancement(eigenvalue_list, 
                                                          eigenbasis_projection_list, 
                                                          num_clock_qubits,
                                                          probability_threshold=probability_threshold,
                                                          exact_alpha=exact_alpha)
    # Construct circuit
    circ = QuantumCircuit(num_clock_qubits+1, name='hybrid_inversion')

    for state, angle in zip(control_state_list, rotation_angle_list):

        gate = RYGate(angle).control(num_ctrl_qubits=num_clock_qubits, label='egn_inv', ctrl_state=state)
        circ.append(gate, circ.qubits)

    return circ

def HybridInversion(eigenvalue_list : list, 
                    eigenbasis_projection_list : list, 
                    num_clock_qubits : int
                    ) -> QuantumCircuit:
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
    # set t0 to reach the highest positive integer for the maximum eigenvalue
    scale = abs((0.5-2**-num_clock_qubits)/abs(max(eigenvalue_list, key=abs)))

    # eigenvalues that pass the first layer of filtering
    eigenvalues = [eigen for i, eigen in enumerate(eigenvalue_list) if 0 < abs(eigen) < eigenbasis_projection_list[i]*(2**num_clock_qubits)]
    control_state_list = [round(value*scale*(2**(num_clock_qubits)-1)) for value in eigenvalues]

    #construct circuit
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


def CanonicalInversion(num_clock_qubits: int) -> QuantumCircuit:
    """
    The function `CannonicalInversion` creates a universally controlled rotation on a
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

def GrayCodeInversion(num_clock_qubits : int):
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
def alpha(delta : float, 
          T : float):
    """
    This Python function calculates the simplified equation 16 in [1].
    
    :param delta: Delta is the distance on the unit circle between the ideal eigenvalue and the control 
    state in question.
    :param T: Parameter determining the total distance between |0> and the maximum control state.
    :return: The real valued amplitude of the entanglement alpha.

    References:
        [1]: Li, X., Phillips, C.,(2024). Detailed Error Analysis for the HHL Algorithm.
        'https://arxiv.org/pdf/2401.17182.pdf'
    """

    coefficient = np.sqrt(2)*np.sin((np.pi)/(2*T))/T
    numerator = abs(np.cos(delta/(2*T))*np.cos(delta/2))
    denominator = abs(np.sin((delta+np.pi)/(2*T))*np.sin((delta-np.pi)/(2*T)))
    return coefficient*numerator/denominator

def Enhancement(eigenvalue_list : list[float],
                eigenbasis_projection_list : list[float], 
                num_clock_qubits : int, 
                probability_threshold : float =0,
                exact_alpha: bool =False,
                ) -> tuple((list[int], list[float])):
    """
    The function `enhanced_angle_processing_practical` uses the classical enhancement to calculate control states 
    and rotation angles of a 'num_clock_qubits' bit eigenvalue inversion circuit based on eigenvalues and 
    eigenbasis projection estimates that have greater bit precision.
    
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
    :return: The function `Enhancement` returns a tuple containing two lists:
    `control_state_list` and `rotation_angle_list`, which are used to determine the control states and rotation angles
    of MCRY gates used in enhanced eigenvalue inversion.
    """
    
    clock = num_clock_qubits
    scale = abs((0.5-2**-clock)/abs(max(eigenvalue_list, key=abs)))
    
    constant = 1 # set initial constant C. Will be scaled based on relevant eigenvaluesS
    
    probability_dictionary = {}

    # Step 1: Determine amplitude of entanglement
    for value, projection in zip(eigenvalue_list, eigenbasis_projection_list):

        # Discard empty eigenvalues. Only relevant if the eigenvalue_list has been calculated classically
        if projection==0:
            continue

        # Calculate the adjacent k bit states and the distance between them
        state = value*scale*(2**(clock)-1)
        state_floor = int(np.floor(state))
        state_ciel = int(np.ceil(state))
        increment = abs(state_ciel-state_floor)

        # Discard 0
        if state_floor == 0 and state_ciel == 0:
            continue

        # If perfectly estimated
        elif increment==0:
            if state_floor < 0:
                state_floor += int(2**clock)

            if state_floor not in probability_dictionary.keys():
                probability_dictionary[state_floor] = {}

            probability_dictionary[state_floor][value] = projection

        else:
            for ctrl_state in [state_floor, state_ciel]:
                if exact_alpha==True:
                    T = 2**clock
                    delta = 2*np.pi * abs(state - ctrl_state) / increment
                    weight = alpha(delta,T)
                else:
                    weight = 1-(abs(state - ctrl_state)/increment)

                # Adjust for two's compliment
                if ctrl_state < 0:
                    ctrl_state += int(2**clock)

                if ctrl_state not in probability_dictionary.keys():
                    probability_dictionary[ctrl_state] = {}

                probability_dictionary[ctrl_state][value] = projection*weight

    amplitude_dictionary = {}
        
    for state, vectors in probability_dictionary.items():
        # See if 0 is a valid control state
        if state == 0:
            all_positive = all(value >= 0 for value in vectors.values())
            all_negative = all(value <= 0 for value in vectors.values())
            if not (all_positive or all_negative):
                continue
    
        final_amplitude = np.average([constant/value for value in vectors.keys()], weights= list(vectors.values()))
        if abs(sum(vectors.values())*final_amplitude) > probability_threshold:
            amplitude_dictionary[state] = final_amplitude

    control_state_list, rotation_angle_list = [], []

    constant = abs(1/max(amplitude_dictionary.values(), key=abs)) # scale amplitude constant
    for state, amplitude in amplitude_dictionary.items():
        control_state_list.append(state)
        rotation_angle_list.append(2*np.arcsin(constant*amplitude))
    
    return control_state_list, rotation_angle_list
