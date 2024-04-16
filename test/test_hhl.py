import __future__

import unittest
import numpy as np
from enhanced_hybrid_hhl import (ExampleQLSP,
                                 HHL)

class TestHHL(unittest.TestCase):
    '''Test HHL'''
    def testCanInversion(self):
        test_problem = ExampleQLSP(0.33)
        can_HHL = HHL(get_result_function="get_fidelity_result")
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_fidelity = abs(can_result.results_processed)
        self.assertTrue(can_fidelity > 0.5)

if __name__ == '__main__':
    unittest.main()