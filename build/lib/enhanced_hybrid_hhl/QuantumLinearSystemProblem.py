from __future__ import annotations
import numpy as np
from typing import Union, List
import pandas as pd
from scipy.stats import unitary_group
from qiskit.quantum_info import random_hermitian
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
import matplotlib.pyplot as plt
from scipy import linalg as la
from qiskit.extensions import HamiltonianGate
from qiskit import QuantumCircuit
from collections.abc import Callable

class QuantumLinearSystemProblem:
    r'''Quantum Linear Systems Problem. Object that either
    A) generates a random hermitian matrix and b vector
    B) validates A matrix and b vector for HHL
    methods then compute the expected result of QPE using A and initial state |b>, as well as what the observed state |x> after HHL'''
    
    def __init__(self,  
                A_matrix: Union[np.ndarray, List[float]] | None = None, 
                b_vector: Union[np.ndarray, List[float]] | None = None,
                post_processing: Union[Callable[[list[float]], list[float]], Callable[[float], float]] | None = None,
                state_preparation: QuantumCircuit | None = None,
                hamiltonian_simulation: QuantumCircuit | None = None):
        
        if A_matrix is not None:
            try:     
                self._A_matrix = np.asmatrix(A_matrix)
            except:        
                raise ValueError('A_Matrix must be ArrayLike')

            if not np.allclose(self._A_matrix, np.conj(self._A_matrix).T):
                raise ValueError('A is not Hermitian')

            if not np.log2(self._A_matrix.shape[0])%1 == 0:
                raise ValueError('A_Matrix dimension is invalid')

            self.num_qubits = round(np.log2(self._A_matrix.shape[0]))
            
        
        elif hamiltonian_simulation is None:
            raise ValueError('A_matrix or hamiltonian_simulation must be provided')
        
        if b_vector is not None:

            try:     
                self._b_vector = np.asmatrix(b_vector)
            except:        
                raise ValueError('b vector must be ArrayLike')

            if self._b_vector.shape[0] == 1:
                self._b_vector = self.b_vector.T

            if not self._b_vector.shape[1]==1:
                raise ValueError('b_vector must be a vector')
            
            if not self._b_vector.shape[0] == self._A_matrix.shape[0]:
                raise ValueError('b_vector must have the same number of rows as A_matrix')
        
        elif state_preparation is None:
            raise ValueError('b_vector or state_preparation must be provided')
        
        
        self._state_preparation = state_preparation
        self._hamiltonian_simulation = hamiltonian_simulation

    
    @property
    def A_matrix(self) -> np.matrix | None:
        r"""Get the matrix :math:`\mathcal{A}` in the linear system problem.

        Returns:
            The :math:`\mathcal{A}` matrix as a numpy matrix.
        """
        return self._A_matrix

    @A_matrix.setter
    def A_matrix(self, A_matrix: np.matrix) -> None:
        r"""Set the :math:`\mathcal{A}` matrix, that encodes the amplitude to be estimated.

        Args:
            A_matrix: The :math:`\mathcal{A}` matrix as a numpy matrix.
        """
        self._A_matrix = A_matrix

    @property
    def b_vector(self) -> np.matrix | None:
        r"""Get the :math:`\mathcal{A}` operator encoding the amplitude :math:`a`.

        Returns:
            The :math:`\mathcal{A}` operator as `QuantumCircuit`.
        """
        return self._b_vector

    @b_vector.setter
    def b_vector(self, b_vector: np.matrix) -> None:
        r"""Set the :math:`\mathcal{A}` operator, that encodes the amplitude to be estimated.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator.
        """
        self._b_vector = b_vector

    @property
    def state_preparation(self) -> QuantumCircuit| None:
        r"""Get the operator to prepare the state of the b vector in the b quantum register
        
        Returns:
            The operator that initializes the :math:`\ket{b}` state as a `QuantumCircuit`."""
        if self._state_preparation is not None:
            return self._state_preparation
        
    @state_preparation.setter
    def state_preparation(self, state_preparation: np.matrix) -> None:
        r"""Set the :math:`\mathcal{A}` operator, that encodes the amplitude to be estimated.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator.
        """
        self._state_prepatation = state_preparation

    @property
    def hamiltonian_simulation(self) -> QuantumCircuit| None:
        r"""Get the operator to prepare the state of the b vector in the b quantum register
        
        Returns:
            The operator that initializes the :math:`\ket{b}` state as a `QuantumCircuit`."""
        return self._hamiltonian_simulation

    @hamiltonian_simulation.setter
    def hamiltonian_simulation(self, hamiltonian_simulation: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{A}` operator, that encodes the amplitude to be estimated.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator.
        """
        self._hamiltonian_simulation = hamiltonian_simulation
        

A = [[1,2], [2,1]]
b = [0,1]       
test = QuantumLinearSystemProblem(A, b)