import numpy as np
import operator
import pandas as pd
from scipy.stats import unitary_group
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PhaseEstimation, QFT
from qiskit.extensions import HamiltonianGate, PhaseGate, Initialize, RYGate
from qiskit.quantum_info import random_hermitian
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
import matplotlib.pyplot as plt
from qiskit.algorithms.phase_estimators import PhaseEstimator
from scipy import linalg as la
import matplotlib as mpl

from QuantumLinearSystemsProblem import QuantumLinearSystemsProblem

class QCL_QPE():
    '''Quantum Phase Estimation - Quantum Conditional Logic outlined here: https://arxiv.org/abs/2110.15958
    This is a phase estimation algorithm that replaces the clock qubits with classical bits. This is used to measure the relative probabilities of various eigenvectors
    in |b>, which is then used to selectively perform the eigenvalue inversion step of HHL on only the important values'''
    def __init__(self,
                 QLSP,
                 clock = 6,
                 max_eigen = None,
                 min_prob = 0.01,
                 max_clock = None
                ) -> None:
        self.problem = QLSP
        self.clock = int(clock)
        self.max_eigen = max_eigen
        self.min_prob = 2**(-clock)
        if max_clock != None:
            self.max_clock = int(max_clock)

    def cust_initialize(self, vector, label):
        '''The initialize instruction takes up a lot of space in the circuit diagram when the initializing vector |b> is large.
        This method creates an instruction that initializes a vector without listing the vector amplitudes in the circuit diagram'''
        init = Initialize(vector)
        circ_i = QuantumCircuit(init.num_qubits)
        circ_i.append(init, circ_i.qubits)
        return circ_i.to_instruction(label=label)
    
    
    def get_results(self, scale):
        '''This method runs the QPE_QCL circuit with the AerSimulator and converts the results from two's complement. 
        The scale determines the time steps of the Hamiltonian simulation operator. '''
        circ = self.construct_circuit(scale)
        from qiskit.providers.aer import AerSimulator
        sim = AerSimulator(method="statevector")
        transp = transpile(circ, sim)
        simulator_result = sim.run(transp, shots=1000).result()
        simulator_counts = simulator_result.get_counts()
        tot = sum(simulator_counts.values())
        return {(int(key,2) if key[0]=='0' else (int(key,2) - (2**(len(key))))) : value / tot for key, value in simulator_counts.items()}

    
    def test_scale(self, scale):
        
        Gamma = scale/(2**self.clock)
        results = self.get_results(Gamma)
        #print('got results')
        abs_eigens = {abs(eig) : prob for eig, prob in results.items() if prob > self.min_prob}
        #print(abs_eigens)
        test = abs_eigens[0]
        #print(test)
        if test>(1-self.min_prob):
            return True
        else:
            return False
    
    def find_scale(self, alpha):
        '''This method combines algorithm 1 and 2 from the paper linked above. Combined these algorithms determine the optimal time 
        parameter of the hamiltonian simulation.'''
        scale = 1/alpha
        over_approximation = self.test_scale(scale)
        while over_approximation == False:
            scale /= 2**(self.clock-1)
            over_approximation = self.test_scale(scale)
        
        x = 0 
        target = int((2**(self.clock-1)-1))
        #print('target =',target)
        while x != target:
            #print(scale)
            results = self.get_results(scale)
            #self.plot_results(results)
            #print('got results')
            eigens = {eig : prob for eig, prob in results.items() if prob > self.min_prob}
            #print(eigens)
            x = abs(max(eigens.keys(), key=abs))
            #print('x=',x)
            
            if not x == 0:
                scale /= x    
            
            scale *= target
            #print(scale)
            
        return scale
    
    def adjust_clock(self):
        '''This method determined te minimum number of bits needed to distinguish the lowest eigenvalue of interest from 0,
        which is the required number of bits for eigenvalue inversion'''
        min_eig = None
        while min_eig == None:
            print(self.clock)
            results = self.get_results(self.scale)
            eigens = {eig : prob for eig, prob in results.items() if prob > self.min_prob}
            #print('got results')
            test = 0
            if 0 in eigens.keys():
                test = eigens[0]
                print('a')
            if test > self.min_prob: #sorted(eigens, key=eigens.get, reverse=True)[:n_eigs]:
                self.clock += 1
                self.min_prob /= 2 
                #print(self.clock
                print('b')
            elif self.clock >= self.max_clock:
                min_eig = 0
                print('c')
            else:
                min_eig = min(eigens.keys(), key=abs)
                print('d')
        return self.clock
    
    def construct_circuit(self, scale):
        '''Constructs QPE_QCL circuit for a given time parameter'''
        clock = self.clock
        circ = QuantumCircuit(self.problem.qubits+1,clock)
        circ.append(self.cust_initialize(list(self.problem.b_state.T[0]),'|b>'), list(range(1, circ.num_qubits)))
        for clbit in range(clock):
            if clbit!=0:
                circ.append(self.cust_initialize([1,0],'|0>'), [0])
            circ.h(0)
            power=2**(clock-clbit-1)
            ham = HamiltonianGate(self.problem.A,-2*np.pi*scale).power(power).control()
            circ.append(ham, circ.qubits)
            
            for i in reversed(range(clbit)):
                #tb = QuantumCircuit(1,1)
                if i < clock:   
                    N = (2**(i+2))
                    #tb.p(np.pi*2/N,0)
                    control = clbit-i-1
                    with circ.if_test((control,1)) as passed:
                        circ.p(-np.pi*2/N,0)
                #circ.if_else((control,1), true_body=tb, false_body=None, qubits=[0], clbits=[control])
            circ.h(0)
            circ.measure(0,[clbit])

        return circ
    
    def estimate(self, alpha=50):
        '''Find the scale and number of clock qubits needed. Returns an estimation of n_egns most important eigenvalues of the eigenvectors needed in to represent |b> in the eigenbasis of A.'''
        if self.max_eigen == None:
            self.scale = self.find_scale(alpha)
        else:
            self.scale = abs((0.5-2**-self.clock)/self.max_eigen)
        
        if hasattr(self, "max_clock"):
            self.adjust_clock()
            self.scale = abs((0.5-2**-self.clock)/self.max_eigen)
                
        results = self.get_results(self.scale)
        self.egn = {eig/(self.scale*2**(self.clock)) : prob for eig, prob in results.items() if prob > self.min_prob}
        return results
    
def plot_results(title, results, ideal, threshold):

        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['font.size'] = 22
        fig = plt.figure()
        values = [key/(qpe.scale*2**(qpe.clock)) for key in results.keys()]
        plt.bar(values, results.values(), zorder=0, width = 0.2, label = 'Statevector Simulator')
        plt.axhline(y=threshold, color = 'r', linestyle = '--', label='Threshold probability')
        plt.grid(visible = True, axis='x')
        
        ideal_threshold = {val : prob for val, prob in ideal.items() if prob > 0}
        plt.scatter(ideal_threshold.keys(), ideal_threshold.values(), zorder=1, label='Reference Values')
        plt.title(title)
        plt.xlabel(r'$\tilde{\lambda}^+_i$')
        plt.ylabel(r'$|\beta_i|^2$')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize = 16)
        return fig.savefig(title+'.png',bbox_inches='tight')