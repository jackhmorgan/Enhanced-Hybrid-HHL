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

class HHL(QuantumCircuit):
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
    
    def construct_circuit(self, scale, clock, approx_degree=0):
        '''This method creates the NISQ_HHL circuit'''
        self.scale = scale
        self.clock = clock
        ham = HamiltonianGate(self.problem.A, -2*np.pi*self.scale)
        #print(self.clock)
        #print(approx_degree)
        iqft = QFT(self.clock, inverse=True, approximation_degree=approx_degree, do_swaps=False).reverse_bits()
        qpe = PhaseEstimation(self.clock, ham, iqft)
        #print(egn)
        #print(clock)
        #egn_inv = self.egn_inv_NISQ(egn)
        egn_inv = ExactReciprocal(self.clock, 2*2**-self.clock, neg_vals=True)
        b_prep = self.cust_initialize(list(self.problem.b_state.T[0]),'|b>')
        
        circ = QuantumCircuit(qpe.num_qubits+1, problem.qubits+1)
        
        circ.append(b_prep, range(-int(problem.qubits), 0))
        circ.append(qpe, range(1, circ.num_qubits))
        circ.append(egn_inv, list(range(self.clock,-1,-1)))
        #circ.append(egn_inv, list(range(self.clock,0,-1))+[0])
        circ.measure(0,0)
        ''' ATTENTION HAMED - we only want to perform the inverse qft and qubit measurement if the first part of the algorithm is successful (i.e. the 0th qubit is |1>'''
        circ.append(qpe.inverse(), range(1, circ.num_qubits))
        circ.measure(range(egn_inv.num_qubits, circ.num_qubits), range(1, circ.num_clbits))
        #with circ.if_test((0,1)) as passed:
        #    circ.append(qpe.inverse(), range(1, circ.num_qubits))
        #    circ.measure(range(egn_inv.num_qubits, circ.num_qubits), range(1, circ.num_clbits))
        #circ.save_statevector()
        return circ
        
        
     
    def estimate_x(self, clock):
        ''' Runs QPE_QCL and uses the results to run NISQ_HHL'''
    
        self.clock = int(clock)
        self.egn = self.problem.ideal_eigen()
        self.scale = abs((0.5-2**-clock)/max(self.egn, key=abs))
        qpe_qcl = QCL_QPE(self.problem, clock=6)
        qpe_qcl.estimate()
        self.egn = qpe_qcl.egn
        #print('scale=',self.scale)
        self.circ = self.construct_circuit(self.egn, self.clock)
        from qiskit.providers.aer import AerSimulator
        sim = AerSimulator(method="statevector")
        transp = transpile(self.circ, sim)
        simulator_result = sim.run(transp, shots=1000).result()
        from qiskit.result import marginal_counts
        #print(simulator_result.get_counts())
        simulator_counts = simulator_result.get_counts()
        #statevector = simulator_result.get_statevector()
        tot = sum(simulator_counts.values())

        return simulator_counts, 0 #statevector #{(int(key,2)-1)/2  : value for key, value in simulator_counts.items()}
