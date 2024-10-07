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
                                 HybridInversion,
                                 CanonicalInversion,
                                 EnhancedHybridInversion,
                                 ExampleQLSP,
                                 QuantumLinearSystemProblem,
                                 QuantumLinearSystemSolver,
                                 HHL)
from qiskit_aer import AerSimulator
from qiskit_ionq import IonQProvider
from qiskit_ibm_runtime import Session, QiskitRuntimeService

class TestHHL(unittest.TestCase):
    '''Test HHL'''
    def testFidelityResult(self):
        test_problem = ExampleQLSP(0.33)
        can_HHL = HHL(get_result_function="get_fidelity_result",
                      eigenvalue_inversion=CanonicalInversion)
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_fidelity = can_result.results_processed
        self.assertTrue(can_fidelity > 0.8)

    def testCircuitDepthResults(self):
        service = QiskitRuntimeService()
        backend = service.backend('ibm_torino')
        test_problem = ExampleQLSP(0.33)
        ideal_x_statevector = QuantumLinearSystemSolver(test_problem).ideal_x_statevector
        can_HHL = HHL(get_result_function="get_circuit_depth_result", 
                      backend = backend)
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1,
                  )
        can_depth = can_result.circuit_depth
        self.assertTrue(can_depth > 50)

        can_HHL = HHL(get_result_function="get_circuit_depth_result_st", 
                      eigenvalue_inversion=CanonicalInversion,
                      backend= backend,
                      statevector= ideal_x_statevector)
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1,
                  )
        can_depth_st = can_result.circuit_depth
        self.assertTrue(can_depth_st > can_depth)

    def testSwapTestResult(self):
        backend = AerSimulator()
        test_problem = ExampleQLSP(0.33)
        ideal_x_statevector = QuantumLinearSystemSolver(test_problem).ideal_x_statevector
        can_HHL = HHL(get_result_function="get_swaptest_result", 
                      backend = backend,
                      statevector = ideal_x_statevector)
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_result = can_result.results_processed
        self.assertTrue(can_result > 0.8)

    def testIonQResult(self):
        ionq_provider = IonQProvider()
        ionq_backend = ionq_provider.get_backend('ionq_simulator')

        test_problem = ExampleQLSP(0.33)
        ideal_x_statevector = QuantumLinearSystemSolver(test_problem).ideal_x_statevector
        can_HHL = HHL(get_result_function="get_ionq_result", 
                      backend = ionq_backend,
                      statevector = ideal_x_statevector)
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1,
                  quantum_conditional_logic=False)
        can_result = can_result.results_processed
        self.assertTrue(can_result > 0.8)
    
    def testSessionResult(self):
        service = QiskitRuntimeService()
        session_backend = service.least_busy(operational=True, simulator=True)
        test_problem = ExampleQLSP(0.33)
        ideal_x_statevector = QuantumLinearSystemSolver(test_problem).ideal_x_statevector
        with Session(service=service, backend=session_backend) as session:
            can_HHL = HHL(get_result_function="get_session_result", 
                        session = session,
                        statevector = ideal_x_statevector,
                        )
            can_result = can_HHL.estimate(problem = test_problem,
                    num_clock_qubits=3,
                    max_eigenvalue=1,
                    quantum_conditional_logic=False)
            can_result = can_result
            self.assertIsInstance(can_result.circuit_depth, int)
            self.assertIsInstance(can_result.job_id, str)


    

if __name__ == '__main__':
    unittest.main()