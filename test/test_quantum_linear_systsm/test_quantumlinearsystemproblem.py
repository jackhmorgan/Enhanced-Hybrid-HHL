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
import unittest
import numpy as np
from enhanced_hybrid_hhl import QuantumLinearSystemProblem

class TestQuantumLinearSystemProblem(unittest.TestCase):
    def test_valid_input(self):
        # Test with valid input
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = np.matrix([[1], [0]])
        problem = QuantumLinearSystemProblem(A_matrix, b_vector)

        # Additional assertions if needed

    def test_non_array_like_A(self):
        # Test with non-NumPy arrays
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = np.matrix([[1], [0]])
        with self.assertRaises(ValueError):
            QuantumLinearSystemProblem('A_matrix', 'b_vector')

    def test_non_hermitian_matrix(self):
        # Test with incompatible shapes
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = np.matrix([[1], [0]])
        with self.assertRaises(ValueError):
            QuantumLinearSystemProblem(np.array([[1, 2], [3, 4]]), 'b_vector')

    def test_invalid_dimensions(self):
        # Test with a non-square matrix
        A_matrix = [[-1.11159804+0.j, -0.28732079-0.42142166j, 0.25517152-0.1358301j],
       [-0.28732079+0.42142166j, -0.20427974+0.j, 0.07023417+0.26861566j],
       [0.25517152+0.1358301j, 0.07023417-0.26861566j, 2.20003348+0.j]]
        b_vector = np.matrix([[1], [0]])
        with self.assertRaises(ValueError):
            QuantumLinearSystemProblem(A_matrix, b_vector)

    def test_no_A_Matrix(self):
        # Test with invalid dimensions
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = np.matrix([[1], [0]])
        with self.assertRaises(ValueError):
            QuantumLinearSystemProblem(b_vector = b_vector)

    def test_non_ArrayLike_b_vector(self):
        # Test with invalid b_vector type
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = 'b_vector'
        with self.assertRaises(ValueError):
            QuantumLinearSystemProblem(A_matrix, b_vector)

    def test_invalid_b_vector_shape(self):
        # Test with invalid b_vector shape
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = np.matrix([[1,2],[3,4]])
        with self.assertRaises(ValueError):
            QuantumLinearSystemProblem(A_matrix, b_vector)
    
    def test_input_dimension_disagreement(self):
        # Test with invalid b_vector shape
        A_matrix = np.matrix([[0.5, -0.25], [-0.25, 0.5]])
        b_vector = np.matrix([[1],[0],[0]])
        with self.assertRaises(ValueError):
            QuantumLinearSystemProblem(A_matrix, b_vector)

if __name__ == '__main__':
    unittest.main()