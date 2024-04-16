from enhanced_hybrid_hhl import QuantumLinearSystemProblem as QLSP
from enhanced_hybrid_hhl import RandomQLSP

import unittest
import numpy as np

class TestQuantumLinearSystemProblem(unittest.TestCase):
    def test_valid_input(self):
        # Test with valid input
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = np.matrix([[1], [0]])
        problem = QLSP(A_matrix, b_vector)

        # Additional assertions if needed

    def test_non_array_like_A(self):
        # Test with non-NumPy arrays
        with self.assertRaises(ValueError):
            QLSP('A_matrix', np.matrix([[1], [0]]))

    def test_non_hermitian_matrix(self):
        # Test with incompatible shapes
        with self.assertRaises(ValueError):
            QLSP(np.array([[1, 2], [3, 4]]), np.matrix([[1], [0]]))

    def test_invalid_dimensions(self):
        # Test with a non-square matrix
        A_3_by_3 = [[-1.11159804+0.j, -0.28732079-0.42142166j, 0.25517152-0.1358301j],
       [-0.28732079+0.42142166j, -0.20427974+0.j, 0.07023417+0.26861566j],
       [0.25517152+0.1358301j, 0.07023417-0.26861566j, 2.20003348+0.j]]
        with self.assertRaises(ValueError):
            QLSP(A_3_by_3, np.matrix([[1], [0], [0]]))

    def test_no_A_Matrix(self):
        # Test with invalid dimensions
        with self.assertRaises(ValueError):
            QLSP(b_vector = np.matrix([[1], [0]]))

    def test_non_ArrayLike_b_vector(self):
        # Test with invalid b_vector type
        with self.assertRaises(ValueError):
            QLSP(np.matrix([[0.5, -0.25], [-0.25, 0.5]]), b_vector = 'b_vector')

    def test_invalid_b_vector_shape(self):
        # Test with invalid b_vector shape
        with self.assertRaises(ValueError):
            QLSP(np.matrix([[0.5, -0.25], [-0.25, 0.5]]), b_vector = np.matrix([[1,2],[3,4]]))
    
    def test_input_dimension_disagreement(self):
        # Test with invalid b_vector shape
        with self.assertRaises(ValueError):
            QLSP(np.matrix([[0.5, -0.25], [-0.25, 0.5]]), b_vector = np.matrix([[1],[0],[0]]))

class TestRandomQLSP(unittest.TestCase):
    def test_random_QLSP(self):
        result = RandomQLSP(2,5.0)
        self.assertIsInstance(result, QLSP)

if __name__ == '__main__':
    unittest.main()