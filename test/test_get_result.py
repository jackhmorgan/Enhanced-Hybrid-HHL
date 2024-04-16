import __future__

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
    @classmethod
    def setUpTests(cls):    
        cls.problem = ExampleQLSP(0.33)
        cls.ideal_x_statevector = QuantumLinearSystemSolver(cls.problem).ideal_x_statevector

        cls.backend = AerSimulator()

        cls.service = QiskitRuntimeService()
        cls.session_backend = cls.service.get_backend('ibmq_qasm_simulator')

        ionq_provider = IonQProvider()
        cls.ionq_backend = ionq_provider.get_backend('ionq_simulator')
    '''Test HHL'''
    def testFidelityResult(self):
        test_problem = self.problem
        can_HHL = HHL(get_result_function="get_fidelity_result",
                      eigenvalue_inversion=CanonicalInversion)
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_fidelity = abs(can_result.results_processed)
        self.assertTrue(can_fidelity > 0.8)

    def testCircuitDepthResults(self):
        can_HHL = HHL(get_result_function="get_circuit_depth_result", 
                      backend = self.backend)
        can_result = can_HHL.estimate(problem = self.problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_depth = abs(can_result.results_processed)
        self.assertTrue(can_depth > 50)

        can_HHL = HHL(get_result_function="get_circuit_depth_result_st", 
                      eigenvalue_inversion=CanonicalInversion,
                      backend= self.backend)
        can_result = can_HHL.estimate(problem = test_problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_depth_st = abs(can_result.results_processed)
        self.assertTrue(can_depth_st > can_depth)

    def testSwapTestResult(self):
        can_HHL = HHL(get_result_function="get_swaptest_result", 
                      backend = self.backend,
                      statevector = self.ideal_x_statevector)
        can_result = can_HHL.estimate(problem = self.problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_result = abs(can_result.results_processed)
        self.assertTrue(can_result > 0.8)

    def testIonQResult(self):
        can_HHL = HHL(get_result_function="get_ionq_result", 
                      backend = self.ionq_backend,
                      statevector = self.ideal_x_statevector)
        can_result = can_HHL.estimate(problem = self.problem,
                  num_clock_qubits=3,
                  max_eigenvalue=1)
        can_result = abs(can_result.results_processed)
        self.assertTrue(can_result > 0.8)
    
    def testSessionResult(self):
        with Session(service=self.service, backend=self.session_backend) as session:
            can_HHL = HHL(get_result_function="get_session_result", 
                        session = session,
                        statevector = self.ideal_x_statevector)
            can_result = can_HHL.estimate(problem = self.problem,
                    num_clock_qubits=3,
                    max_eigenvalue=1)
            can_result = abs(can_result.results_processed)
            self.assertTrue(can_result > 0.8)


    

if __name__ == '__main__':
    unittest.main()
        