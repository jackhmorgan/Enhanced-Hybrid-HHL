import __future__

from HHL import HHL
from enhanced_hybrid_hhl.quantum_linear_system.QuantumLinearSystemSolver import QuantumLinearSystemSolver
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import PhaseEstimation, HamiltonianGate, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_algorithms import AlgorithmError
import numpy as np

def ideal_preprocessing(problem):
    '''Classically computes the ideal eigenvalue list and eigenbasis projection of the QuantumLinearSystemProblem'''

    solution = QuantumLinearSystemSolver(problem=problem) # Classically solves the linear system problem

    eigenvalue_list = solution.eigenvalue_list
    eigenbasis_projection_list = solution.eigenbasis_projection_list

    return eigenvalue_list, eigenbasis_projection_list

class QCL_QPE_IBM():

    def __init__(self,
                 clock,
                 backend,
                 alpha = 50,
                 max_eigenvalue = None,
                 min_prob = None):
        self.clock = clock
        self.backend = backend
        self.alpha = alpha
        self.max_eigenvalue = max_eigenvalue
        if min_prob == None:
           min_prob = 2**-clock
        self.min_prob = min_prob


    def get_result(self, scale):
        '''This method runs the QPE_QCL circuit with the specified backendand converts the results from two's complement. 
        The scale determines the time steps of the Hamiltonian simulation operator. '''
        circ = self.construct_circuit(scale)
        backend = self.backend
        transp = transpile(circ, backend)
        result = backend.run(transp, shots=4000).result()
        counts = result.get_counts()
        tot = sum(counts.values())
        result_dict = {(int(key,2) if key[0]=='0' else (int(key,2) - (2**(len(key))))) : value / tot for key, value in counts.items()}
        self.result = result_dict
        return result_dict

    
    def test_scale(self, scale):
        
        Gamma = scale/(2**self.clock)
        results = self.get_results(Gamma)
        abs_eigens = {abs(eig) : prob for eig, prob in results.items() if prob > self.min_prob}
        test = abs_eigens[0]
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
        
        while x != target:
            
            results = self.get_results(scale)
            eigens = {eig : prob for eig, prob in results.items() if prob > self.min_prob}  
            x = abs(max(eigens.keys(), key=abs))
            
            if not x == 0:
                scale /= x    
            
            scale *= target
            
        return scale
    
    def adjust_clock(self):
        '''This method determined the minimum number of bits needed to distinguish the lowest eigenvalue of interest from 0,
        which is the required number of bits for eigenvalue inversion'''
        min_eig = None
        while min_eig == None:
            results = self.get_results(self.scale)
            eigens = {eig : prob for eig, prob in results.items() if prob > self.min_prob}
            test = 0
            if 0 in eigens.keys():
                test = eigens[0]
            if test > self.min_prob: 
                self.clock += 1
                self.min_prob /= 2 
            elif self.clock >= self.max_clock:
                min_eig = 0
            else:
                min_eig = min(eigens.keys(), key=abs)
    
        return self.clock
    
    def construct_circuit(self, hamiltonian_gate, state_preparation):
        '''Constructs QCL_QPE circuit for a given hamiltonian gate'''
        
        circ = QuantumCircuit(hamiltonian_gate.num_qubits+1, self.clock)
        circ.prepare_state(self.state_prepatation, list(range(1, circ.num_qubits)))
        for clbit in range(self.clock):
            if clbit!=0:
                circ.initialize([1,0],[0])
            circ.h(0)
            power=2**(self.clock-clbit-1)
            ham = self.hamiltonian_simulation.power(power).control()
            circ.append(ham, circ.qubits)
            
            for i in reversed(range(clbit)):
                
                if i < self.clock:   
                    N = (2**(i+2))
                    
                    control = clbit-i-1
                    with circ.if_test((control,1)) as passed:
                        circ.p(-np.pi*2/N,0)
            circ.h(0)
            circ.measure(0,[clbit])
        return circ
    
    def estimate(self, problem):
        '''Find the scale and number of clock qubits needed. Returns an estimation of n_egns most important eigenvalues of the eigenvectors needed in to represent |b> in the eigenbasis of A.'''
        # If the hamiltonian simulation is not specified in the problem, use the standard HamiltonianGate
        if getattr(problem, 'hamiltonian_simulation', None) is None:
            if self.max_eigenvalue == None:
                raise AlgorithmError('An upper bound on the eigenvalues is needed')
            
            scale = abs((0.5-2**-self.clock)/self.max_eigenvalue)
            self.hamiltonian_simulation = HamiltonianGate(problem.A_matrix, -2*np.pi*scale)
        
        else:
            self.hamiltonian_simulation = problem.hamiltonian_simulation
        

        # If the state_preparation is not specified in the problem, use the standard StatePreparation
        
        if getattr(problem, 'state_preparation', None) is None:
            
            self.state_preparation = StatePreparation(Statevector(problem.b_vector))
        
        else:
            self.state_preparation = problem.state_preparation


        if self.max_eigen == None:
            self.scale = self.find_scale(self.alpha)
        else:
            self.scale = abs((0.5-2**-self.clock)/self.max_eigen)
        
        if hasattr(self, "max_clock"):
            self.adjust_clock()
            self.scale = abs((0.5-2**-self.clock)/self.max_eigen)
            
        if not hasattr(self, "result"):
            self.get_results(self.scale)
                
        eigenvalue_list = [eig/(self.scale*2**(self.clock)) for eig in self.result.keys() if self.result[eig] > self.min_prob]
        eigenbasis_projection_list = [self.result[eigen] for eigen in eigenvalue_list]
        
        return eigenvalue_list, eigenbasis_projection_list
    
class QPE_preprocessing():
    def __init__(self,
                 num_eval_qubits,
                 get_result):
        self.num_eval_qubits = num_eval_qubits
        self.get_result = get_result

    def construct_circuit(self, 
                          hamiltonian_simulation,
                          state_preparation):
        circuit = QuantumCircuit(self.num_eval_qubits+hamiltonian_simulation.num_qubits, self.num_eval_qubits)
        circuit.append(state_preparation, range(self.num_eval_qubits, circuit.num_qubits))
        circuit.append(PhaseEstimation(self.num_eval_qubits, hamiltonian_simulation), circuit.qubits)
        circuit.measure(range(self.num_eval_qubits), range(self.num_eval_qubits))
        return circuit
    
    def estimate(self, problem):

        if getattr(problem, 'hamiltonian_simulation', None) is None:
            if self.max_eigenvalue == None:
                raise AlgorithmError('An upper bound on the eigenvalues is needed')
            
            scale = abs((0.5-2**-self.num_clock_qubits)/self.max_eigenvalue)
            hamiltonian_simulation = HamiltonianGate(problem.A_matrix, -2*np.pi*scale)
        
        else:
            hamiltonian_simulation = problem.hamiltonian_simulation
        

        # If the state_preparation is not specified in the problem, use the standard StatePreparation
        
        if getattr(problem, 'state_preparation', None) is None:
            
            state_preparation = StatePreparation(Statevector(problem.b_vector))
        
        else:
            state_preparation = problem.state_preparation

        circ = self.construct_circuit(hamiltonian_simulation, state_preparation)
        result = self.get_result(circ)