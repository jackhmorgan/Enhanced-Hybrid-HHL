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
from enhanced_hybrid_hhl import (ideal_preprocessing,
                                 Yalovetzky_preprocessing,
                                 Lee_preprocessing,
                                 Iterative_QPE_Preprocessing,
                                 EnhancedHybridInversion,
                                 ExampleQLSP,
                                 HHL)
from qiskit_aer import AerSimulator

class TestInversionCircuits(unittest.TestCase):
    '''Test Inversion Circuits'''
    def testIdealPreprocessing(self):
        test_problem = ExampleQLSP(0.33)
        E_H_HHL = HHL(get_result_function="get_fidelity_result",
                      eigenvalue_inversion=EnhancedHybridInversion,
                      preprocessing=ideal_preprocessing)
        result = E_H_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        fidelity = abs(result.results_processed)
        self.assertTrue(fidelity > 0.5)

    def testIterativePreprocessing(self):
        test_problem = ExampleQLSP(0.33)
        backend = AerSimulator()
        preprocessing = Iterative_QPE_Preprocessing(clock=5,
                                                    backend=backend).estimate
        E_H_HHL = HHL(get_result_function="get_fidelity_result",
                      eigenvalue_inversion=EnhancedHybridInversion,
                      preprocessing=preprocessing)
        result = E_H_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        fidelity = abs(result.results_processed)
        self.assertTrue(fidelity > 0.5)

    def testLeePreprocessing(self):
        test_problem = ExampleQLSP(0.33)
        backend = AerSimulator()
        preprocessing = Lee_preprocessing(num_eval_qubits=5,
                                          max_eigenvalue=1,
                                          backend = backend).estimate
        E_H_HHL = HHL(get_result_function="get_fidelity_result",
                      eigenvalue_inversion=EnhancedHybridInversion,
                      preprocessing=preprocessing)
        result = E_H_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        fidelity = abs(result.results_processed)
        self.assertTrue(fidelity > 0.5)

    def testYalovetzkyPreprocessing(self):
        test_problem = ExampleQLSP(0.33)
        backend = AerSimulator()
        preprocessing = Yalovetzky_preprocessing(clock=5,
                                                 backend=backend).estimate
        E_H_HHL = HHL(get_result_function="get_fidelity_result",
                      eigenvalue_inversion=EnhancedHybridInversion,
                      preprocessing=preprocessing)
        result = E_H_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        fidelity = abs(result.results_processed)
        self.assertTrue(fidelity > 0.5)

if __name__ == '__main__':
    unittest.main()
        