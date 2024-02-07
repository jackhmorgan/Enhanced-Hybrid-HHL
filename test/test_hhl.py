import __future__

import sys
import os

# Assuming the parent directory of the tests folder is in your project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import unittest
import numpy as np
from .get_result import get_fidelity_result
from .eigenvalue_preprocessing import ideal_preprocessing
from .inversion_circuits import HybridInversion
from .QuantumLinearSystemProblem import QuantumLinearSystemProblem
from ddt import data
from HHL import HHL

class TestHHL(unittest.TestCase):
    '''Test HHL'''
    @data(QuantumLinearSystemProblem(A_matrix= [[1,-0.33],[-0.33,1]], b_vector = [1,0]))
    def testCannHHL(self, test_problem):
        cann_HHL = HHL(get_fidelity_result)
        cann_result = cann_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        cann_fidelity = abs(cann_result.result_processed)
        self.assertTrue(cann_fidelity > 0.8)

    def test_Hybrid_HHL_gt_Cann_HHL(self, test_problem):

        cann_HHL = HHL(get_fidelity_result)
        cann_result = cann_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        cann_fidelity = abs(cann_result.result_processed)

        self.assertGreaterEqual(cann_fidelity, 0.8)

        E_Hybrid_HHL = HHL(get_result=get_fidelity_result,
                    pre_processing=ideal_preprocessing,
                    eigenvalue_inversion=HybridInversion)
        
        hybrid_result = E_Hybrid_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        
        hybrid_fidelity = abs(hybrid_result.result_processed)

        self.assertGreaterEqual(hybrid_fidelity, cann_fidelity)

if __name__ == '__main__':
    unittest.main()
        