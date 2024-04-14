import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')
from qiskit_aer import AerSimulator
import numpy as np

from enhanced_hybrid_hhl import (HHL, 
                                 EnhancedHybridInversion,
                                 Lee_preprocessing,
                                 Yalovetsky_preprocessing,
                                 ideal_preprocessing,
                                 QuantumLinearSystemProblem,
                                 ExampleQLSP,
                                 QuantumLinearSystemSolver)
clock = 3
threshold = 2**-clock
backend = AerSimulator()
noise_model=None

preprocessing = Lee_preprocessing(clock+2, 1, backend=backend, noise_model=None).estimate

e_h_hhl = HHL(get_result_function='get_fidelity_result',
                        pre_processing=preprocessing,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        probability_threshold = threshold,
                        exact_alpha=True)

exact_results = []
approx_results = []
for lam in [3/24,4/24,5/24,6/24,7/24,8/24,9/24, 10/24, 11/24]:
    problem = ExampleQLSP(lam)
    exact_alpha_result = e_h_hhl.estimate(problem=problem,
                             num_clock_qubits=clock,
                             max_eigenvalue=1,
                             exact_alpha= True,
                             probability_threshold = threshold)
    approx_alpha_result = e_h_hhl.estimate(problem=problem,
                             num_clock_qubits=clock,
                             max_eigenvalue=1,
                             exact_alpha= False,
                             probability_threshold = threshold)
    exact_results.append(exact_alpha_result.results_processed)
    approx_results.append(approx_alpha_result.results_processed)
exact_errors = [np.sqrt(2*(1-fidelity)) for fidelity in exact_results]
print('ave exact error',np.average(exact_errors))
approx_errors = [np.sqrt(2*(1-fidelity)) for fidelity in exact_results]
print('ave approx error', np.mean(approx_errors))
print(exact_errors)
print(approx_errors)