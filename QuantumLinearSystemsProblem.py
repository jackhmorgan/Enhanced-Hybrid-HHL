import numpy as np
import operator
import pandas as pd
from scipy.stats import unitary_group
from qiskit.quantum_info import random_hermitian
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
import matplotlib.pyplot as plt
from scipy import linalg as la

class QuantumLinearSystemsProblem:
    '''Quantum Linear Systems Problem. Object that either:
    A) generates a random hermitian matrix and b vector
    B) validates A matrix and b vector for HHL
    methods then compute the expected result of QPE using A and initial state |b>, as well as what the observed state |x> after HHL'''
    
    def ideal_eigen(self):
        A_eigen = np.linalg.eigh(self.A)
        b_eigen = np.linalg.inv(A_eigen[1]).dot(self.b_state)
        eigen = {}
        for i, amp in enumerate(b_eigen):
            eigen[A_eigen[0][i]] = abs(amp*np.conj(amp))[0,0]
        
        return {eig : prob for eig, prob in eigen.items()}
    
    def ideal_x(self):
        x = np.linalg.solve(self.A, self.b_state)
        x_norm = np.linalg.norm(x)
        x_state = x/x_norm
        x_meas = []
        for amp in x_state[:,0]:
            x_meas.append(abs(amp*np.conj(amp)))
        return (x_state, x_meas)
    
    def __init__(self, 
                qubits=None, 
                A=None, 
                b=None, 
                block_encoder=None):
        size=None
        if qubits != None:
            size = 2**qubits
            self.qubits = qubits
        
        if isinstance(A, (np.ndarray, list)):
            if block_encoder != None:
                self.A = np.asmatrix(block_encoder(A))    
            else:
                self.A = np.asmatrix(A)
                
            self.size = self.A.shape[0]
            if size != self.size and size != None:
                raise Exception('Matrix does not agree with size')
            if self.A.shape[0] != self.A.shape[1] or len(self.A.shape) != 2:
                raise Exception('Matrix size is invalid')
        else: 
            try:
                self.size = int(size)
                self.A = np.asmatrix(random_hermitian(self.size).data)
            except:
                Exception('Size or matrix inputs must be valid')
        
        if np.log2(self.size) % 1 != 0:
            raise Exception('size must be a power of 2')

        self.qubits = round(np.log2(self.size))
        
        if (self.A != self.A.H).all():
            raise Exception('A is not Hermitian')
        
        if b is not None:
            self.b = np.asarray(b)
            if not (self.b.size == self.size or len(self.b.shape)==1):
                raise Exception('b must be a vector')
            
        else:
            self.b = np.asarray(np.random.rand(size)).reshape((size,1))
            
        b_norm = np.linalg.norm(self.b) 
        self.b_state = self.b/b_norm
        self.b_state