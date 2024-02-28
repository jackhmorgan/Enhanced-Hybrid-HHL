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

from __future__ import annotations
import numpy as np
from typing import Union, List

from qiskit import QuantumCircuit
from collections.abc import Callable
from qiskit_algorithms import AlgorithmResult
from qiskit.quantum_info import Statevector

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
            
            self._b_vector = self._b_vector/np.linalg.norm(self._b_vector)
        
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
        

class HHL_Result(AlgorithmResult):
    r"""The results object for amplitude estimation algorithms."""

    def __init__(self) -> None:
        super().__init__()
        self._circuit_results: Union[list[dict[str, int]], dict[str, int], None] = None,
        self._shots: Union[int, None] = None
        self._results_processed: Union[int, None] = None
        self._eigenvalue_list: Union[list[float], None] = None
        self._eigenbasis_projection_list: Union[list[float], None] = None
        self._control_state_list: Union[list[int], None] = None
        self._rotation_angle_list: Union[list[float], None] = None
        self._post_processing: Union[Callable[[float], float], None] = None
        self._ideal_x_statevector: Union[Statevector, None] = None

    @property
    def circuit_results(self) -> Union[list[dict[str, int]], dict[str, int], None]:
        """Return the circuit results. Can be a statevector or counts dictionary."""
        return self._circuit_results

    @circuit_results.setter
    def circuit_results(self, value: Union[list[dict[str, int]], dict[str, int]]) -> None:
        """Set the circuit results."""
        self._circuit_results = value

    @property
    def shots(self) -> int:
        """Return the number of shots used. Is 1 for statevector-based simulations."""
        return self._shots

    @shots.setter
    def shots(self, value: int) -> None:
        """Set the number of shots used."""
        self._shots = value

    @property
    def eigenvalue_list(self) -> list[float]:
        r"""Return the list or relevant eigenvalues of the Quantum Linear System."""
        return self._eigenvalue_list

    @eigenvalue_list.setter
    def eigenvalue_list(self, value: list[float]) -> None:
        r"""Set the list or relevant eigenvalues of the Quantum Linear System."""
        self._eigenvalue_list = value

    @property
    def results_processed(self) -> float:
        """Return the results of the observation after the post-processing has been applied."""
        return self._estimation_processed

    @results_processed.setter
    def results_processed(self, value: float) -> None:
        """Set the results of the observation after the post-processing has been applied."""
        self._estimation_processed = value

    @property
    def eigenbasis_projection_list(self) -> list[int]:
        """Return list of the projestions of :math:`\ket{b}` onto the eigenbasis of :math:`\mathcal{A}`."""
        return self._eigenbasis_projection_list

    @eigenbasis_projection_list.setter
    def eigenbasis_projection_list(self, value: list[int]) -> None:
        """Set the number of Grover oracle queries."""
        self._eigenbasis_projection_list = value

    @property
    def post_processing(self) -> Callable[[float], float]:
        """Return a handle to the post processing function."""
        return self._post_processing

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[float], float]) -> None:
        """Set a handle to the post processing function."""
        self._post_processing = post_processing

    @property
    def control_state_list(self) -> list[int]:
        """Return the control states used for eigenvalue inversion."""
        return self._confidence_interval

    @control_state_list.setter
    def control_state_list(self, control_state_list: list[int]) -> None:
        """Set the control states used for eigenvalue inversion."""
        self._control_state_list = control_state_list

    @property
    def rotation_angle_list(self) -> list[float]:
        """Return the rotation angles used for eigenvalue inversion."""
        return self._control_state_list

    @rotation_angle_list.setter
    def rotation_angle_list(self, rotation_angle_list: list[float]) -> None:
        """Set the post-processed confidence interval (95% interval by default)."""
        self._rotation_angle_list = rotation_angle_list

    @property
    def ideal_x_statevector(self) -> Statevector:
        """Return the classically computed statevector of the solution."""
        return self._ideal_x_statevector

    @ideal_x_statevector.setter
    def ideal_x_statevector(self, ideal_x_statevector: Statevector) -> None:
        """Set the classically computed statevector of the solution."""
        self._ideal_x_statevector = ideal_x_statevector
