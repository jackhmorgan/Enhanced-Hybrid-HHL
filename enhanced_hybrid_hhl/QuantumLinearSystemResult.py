from qiskit_algorithms import AlgorithmResult
from qiskit.quantum_info import Statevector
from typing import Callable, Union

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
    def eigenvalue_projection_list(self) -> list[int]:
        """Return list of the projestions of :math:`\ket{b}` onto the eigenbasis of :math:`\mathcal{A}`."""
        return self._eigenvalue_projection_list

    @eigenvalue_projection_list.setter
    def eigenvalue_projection_list(self, value: list[int]) -> None:
        """Set the number of Grover oracle queries."""
        self._num_oracle_queries = value

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