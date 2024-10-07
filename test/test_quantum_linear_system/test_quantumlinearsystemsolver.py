import unittest
import numpy as np
from ddt import ddt, data

# Import your function and classes here
from enhanced_hybrid_hhl import QuantumLinearSystemSolver, QuantumLinearSystemProblem, HHL_Result
from qiskit.quantum_info import Statevector

@ddt
class QuantumLinearSystemSolverTest(unittest.TestCase):
    
    @data(
        (np.array([[1, 0.3], [0.3, 1]]), np.array([[1], [0]]), Statevector([ 0.95782629+0.j, -0.28734789+0.j])),  # Example 1
        (np.array([[1, -0.5], [-0.5, 1]]), np.array([[1], [0]]), Statevector([0.89442719+0.j, 0.4472136 +0.j])), # Example 2
    )
    def test_solver_with_valid_problem(self, input_data):
        A_matrix, b_vector, x_statevector = input_data
        problem = QuantumLinearSystemProblem(A_matrix, b_vector)
        result = QuantumLinearSystemSolver(problem)
        self.assertIsInstance(result, HHL_Result)
        self.assertTrue(result.ideal_x_statevector.equiv(x_statevector))
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
