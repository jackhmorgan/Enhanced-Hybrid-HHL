import numpy as np
from AssetPricing import Generate_D_Minus_E_problem
from enhanced_hybrid_hhl import HHL, ideal_preprocessing
from qiskit.quantum_info import Operator

utility_function = 'IES'
gamma = 2

problem = Generate_D_Minus_E_problem(utility_function=utility_function,
                                     gamma=gamma,
                                     size=4)
max_eigenvalue = max(ideal_preprocessing(problem=problem)[0], key=abs)

data = np.load(utility_function+'_'+str(gamma)+'tasks_3_4.npz', allow_pickle=True)

classical_operators=data['classical_operators'] 
quantum_upper_bound_operators=data['quantum_upper_bound_operators']
quantum_lower_bound_operators=data['quantum_lower_bound_operators']
benchmark_model=data['benchmark_model']

classical_utilities = []
upper_bounds = []
lower_bounds = []

for i in range(len(classical_operators)):
    # Classical
    classical_operator = Operator(classical_operators[i])
    classical_hhl = HHL(get_result_function='get_simulator_result',
                        operator=classical_operator)
    
    # Quantum Lower Bound
    quantum_lower_bound_operator = Operator(quantum_lower_bound_operators[i])

    quantum_lower_bound_hhl = HHL(get_result_function='get_simulator_result',
                        operator=quantum_lower_bound_operator)
    

    # Quantum Upper Bound
    quantum_upper_bound_operator = Operator(quantum_upper_bound_operators[i])

    quantum_upper_bound_hhl = HHL(get_result_function='get_simulator_result',
                        operator=quantum_upper_bound_operator)
    
    # Run circuits
    classical_utility = classical_hhl.estimate(problem=problem, max_eigenvalue=max_eigenvalue).results_processed
    quantum_lower_bound = quantum_lower_bound_hhl.estimate(problem=problem, max_eigenvalue=max_eigenvalue).results_processed
    quantum_upper_bound = quantum_upper_bound_hhl.estimate(problem=problem, max_eigenvalue=max_eigenvalue).results_processed
    
    # Add to lists
    classical_utilities.append(classical_utility)
    lower_bounds.append(quantum_lower_bound)
    upper_bounds.append(quantum_upper_bound)

print(classical_utilities)
print(lower_bounds)
print(upper_bounds)
