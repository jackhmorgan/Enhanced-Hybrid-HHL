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