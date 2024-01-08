# Imports for Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-ncsu/nc-state/financial-quantu',
)

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
import numpy as np

# Imports that I use
import operator
import pandas as pd
from scipy.stats import unitary_group
from qiskit.circuit.library import PhaseEstimation, QFT
from qiskit.extensions import HamiltonianGate, PhaseGate, Initialize, RYGate
from qiskit.quantum_info import random_hermitian
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
import matplotlib.pyplot as plt
from qiskit.algorithms.phase_estimators import PhaseEstimator
from scipy import linalg as la

from qiskit.circuit.library import ExactReciprocal
from QuantumLinearSystemsProblem import QuantumLinearSystemsProblem as QLSP
from QCL_QPE import QCL_QPE

class E_Hybrid_HHL(QuantumCircuit):
    '''Performs HHL by only inverting over a subset of relevant eigenvalues, as determined by QPE_QCL'''
    def __init__(self,
                problem = None,
                qubits=2, 
                A=None, 
                b=None, 
                block_encoder=None):
        self.problem=problem
        if self.problem==None:
            self.problem = QLSP(qubits, A, b, block_encoder)
        
        
    #def ctrl_state_calculator(self, egn, clock):
        
        
    def egn_inv_NISQ(self, egn):
        '''Sub circuit that performs eigenvalue inversion. For each eigenvalue lambda in egn, rotate the 0th qubit such that the |1> amplitude is C/lambda controlled on the lambda state of the clock register'''
        clock = self.clock
        scale = self.scale
        #print('scale =',scale)
        #print(egn)
        
        constant = abs(min(egn, key=abs))
        
        probability_dictionary = {}
        final_amplitude_dictionary = {}
        
        for value, prob in egn.items():
            state = value*scale*(2**(clock)-1)
            #print(state)
            #print(scale,'scale')
            final_amplitude = (1/value)

            #if (value < 0) and (round(state) != 0):
                #print('value =',value)
                #print('state =', state)
                #state+= int(2**clock)
                
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

                    probability_dictionary[ctrl_state][final_amplitude] = (prob**0.5)*weight

        #print(probability_dictionary)
        
        amplitude_dictionary = {}
        probability_threshold = 0 #2**-(clock+1)
        noise = 0 #2**-clock
        for state, vectors in probability_dictionary.items():
            final_amplitude = np.average(list(vectors.keys()), weights = list(vectors.values()))
            if sum(vectors.values()) > (probability_threshold*final_amplitude+noise):
                amplitude_dictionary[state] = final_amplitude

        #print('--------')
        #print(amplitude_dictionary)
        #print('--------')
        rotation_dictionary={}
        constant = 1/max(amplitude_dictionary.values(), key=abs)
        for state, amplitude in amplitude_dictionary.items():
            rotation_dictionary[state] = 2*np.arcsin(constant*amplitude)
            
        print(rotation_dictionary)
    
        circ_i = QuantumCircuit(clock+1)

        for state, angle in rotation_dictionary.items(): 

            gate = RYGate(angle).control(num_ctrl_qubits=clock, label='egn_inv', ctrl_state=state)
            circ_i.append(gate, circ_i.qubits)

        return circ_i.to_gate()
    
    def cust_initialize(self, vector, label):
        '''The initialize instruction takes up a lot of space in the circuit diagram when the initializing vector |b> is large.
        This method creates an instruction that initializes a vector without listing the vector amplitudes in the circuit diagram'''
        init = Initialize(vector)
        circ_i = QuantumCircuit(init.num_qubits)
        circ_i.append(init, circ_i.qubits)
        return circ_i.to_instruction(label=label)
    
    def construct_circuit(self, egn, clock, max_eigen, statevector=None, approx_degree=0):
        '''This method creates the NISQ_HHL circuit'''
        self.scale = abs((0.5-2**-clock)/max_eigen)
        self.egn = egn
        self.clock = clock
        ham = HamiltonianGate(self.problem.A, -2*np.pi*self.scale)
        #print(self.clock)
        #print(approx_degree)
        iqft = QFT(clock, inverse=True, approximation_degree=approx_degree, do_swaps=False).reverse_bits()
        qpe = PhaseEstimation(clock, ham, iqft)
        #print(egn)
        #print(clock)
        egn_inv = self.egn_inv_NISQ(egn)
        #egn_inv = ExactReciprocal(self.clock, 2*2**-self.clock, neg_vals=True)
        b_prep = self.cust_initialize(list(self.problem.b_state.T[0]),'|b>')
        
        circ = QuantumCircuit(qpe.num_qubits+1, self.problem.qubits+1)
        circ.prepare_state(state=list(self.problem.b_state.T[0]), qubits=range(-int(self.problem.qubits),0), label = '|b>')
        #circ.append(b_prep, range(-int(self.problem.qubits), 0))
        circ.append(qpe, range(1, circ.num_qubits))
        circ.append(egn_inv, list(range(clock,-1,-1)))
        #circ.append(egn_inv, list(range(self.clock,0,-1))+[0])
        circ.measure(0,0)
        ''' ATTENTION HAMED - we only want to perform the inverse qft and qubit measurement if the first part of the algorithm is successful (i.e. the 0th qubit is |1>'''
        circ.append(qpe.inverse(), range(1, circ.num_qubits))
        circ.measure(range(egn_inv.num_qubits, circ.num_qubits), range(1, circ.num_clbits))
        #with circ.if_test((0,1)) as passed:
        #    circ.append(qpe.inverse(), range(1, circ.num_qubits))
        #    if not statevector==None:
        #        circ.save_statevector()
        #    circ.measure(range(egn_inv.num_qubits, circ.num_qubits), range(1, circ.num_clbits))
        return circ
        
        
     
    def estimate_x(self, clock, max_eigen=None, operator=None, statevector=None):
        ''' Runs QPE_QCL and uses the results to run NISQ_HHL'''
    
        self.clock = int(clock)
        qpe_qcl = QCL_QPE(self.problem, clock=6, max_eigen=max_eigen)
        qpe_qcl.estimate()
        self.egn = qpe_qcl.egn
        if max_eigen==None:
            max_eigen = max(self.egn, key=abs)
        #print('scale=',self.scale)
        self.circ = self.construct_circuit(self.egn, self.clock, max_eigen, statevector=statevector)
        from qiskit.providers.aer import AerSimulator
        sim = AerSimulator(method="statevector")
        transp = transpile(self.circ, sim)
        simulator_result = sim.run(transp, shots=1000).result()
        from qiskit.result import marginal_counts
        #print(simulator_result.get_counts())
        if statevector == None:
            simulator_counts = simulator_result.get_counts()
            return simulator_counts
        else:
            statevector = simulator_result.get_statevector()
            return statevector
