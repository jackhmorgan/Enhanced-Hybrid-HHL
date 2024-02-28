import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')

from enhanced_hybrid_hhl import (ExampleQLSP,
                                 RandomQLSP,
                                 QuantumLinearSystemProblem,
                                 Lee_preprocessing,
                                 ideal_preprocessing)
from qiskit_aer import AerSimulator
import numpy as np
simulator = AerSimulator()
test_eigenvalues = [-1, -5/7, -2/7, 1/4, 1/3, 11/32, 3/4, 1]
problem_A_matrix = np.diag(test_eigenvalues)
problem_b_vector = np.array([[1.], [0.], [0], [0], [0], [1], [0], [0]])/np.sqrt(2)

random_problem = QuantumLinearSystemProblem(problem_A_matrix, problem_b_vector)
four_preprocessing=Lee_preprocessing(num_eval_qubits=4, 
                                  backend=simulator,
                                  max_eigenvalue=1)


aaa_four_result = four_preprocessing.estimate(random_problem)[0]

six_preprocessing=Lee_preprocessing(num_eval_qubits=6, 
                                  backend=simulator,
                                  max_eigenvalue=1)


aaa_six_result = six_preprocessing.estimate(random_problem)[0]

for output_num in output_list:
        # Check if output_num is close to any number in the solution list
        close_to_any = any(np.isclose(output_num, solution_list, atol=tolerance))
        
        # If output_num is not close to any number in the solution list, fail the test
        assert close_to_any