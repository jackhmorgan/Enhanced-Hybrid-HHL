import numpy as np
from qiskit import QuantumCircuit
from qiskit.extensions import RYGate

def HybridInversion(eigenvalue_list, eigenbasis_projection_list, num_clock_qubits) -> QuantumCircuit:
    
    control_state_list, rotation_angle_list = enhanced_angle_processing_practical(eigenvalue_list, eigenbasis_projection_list, num_clock_qubits)
    circ = QuantumCircuit(num_clock_qubits+1, name='hybrid_inversion')

    for state, angle in zip(control_state_list, rotation_angle_list):

        gate = RYGate(angle).control(num_ctrl_qubits=num_clock_qubits, label='egn_inv', ctrl_state=state)
        circ.append(gate, circ.qubits)

    return circ

def CannonicalInversionFunction(eigenvalue_list, eigenbasis_projection_list, num_clock_qubits):
    from qiskit.circuit.library import ExactReciprocal
    er = ExactReciprocal(num_clock_qubits, 2*2**-num_clock_qubits, neg_vals=True)
    return er

def enhanced_angle_processing_practical(eigenvalue_list, 
                                        eigenbasis_projection_list, 
                                        num_clock_qubits, 
                                        probability_threshold=0) -> tuple((list[int], list[float])):
    
    clock = num_clock_qubits
    scale = abs((0.5-2**-clock)/abs(max(eigenvalue_list, key=abs)))
    
    constant = abs(min(eigenvalue_list, key=abs))
    
    probability_dictionary = {}
    final_amplitude_dictionary = {}

    for value, projection in zip(eigenvalue_list, eigenbasis_projection_list):

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
        if sum(vectors.values())*final_amplitude > probability_threshold:
            amplitude_dictionary[state] = final_amplitude

    control_state_list, rotation_angle_list = [], []

    constant = abs(1/max(amplitude_dictionary.values(), key=abs))
    for state, amplitude in amplitude_dictionary.items():
        control_state_list.append(state)
        rotation_angle_list.append(2*np.arcsin(constant*amplitude))
    
    return control_state_list, rotation_angle_list