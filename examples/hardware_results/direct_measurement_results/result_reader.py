import json
import os
import numpy as np

from enhanced_hybrid_hhl import QuantumLinearSystemSolver, QuantumLinearSystemProblem

#file_name = 'iqm_deneb_small_matrix_hhl3.json'
#file_name = 'nst_aria1_small_matrix_hhl1.json'
file_name = 'nst_aria1_small_matrix_hhl1.json'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

problem_list = [QuantumLinearSystemProblem(A_matrix=problem, b_vector=[1,0]) for problem in json_data['problem_list']]

canonical_results = json_data['canonical_results']
canonical_fidelities = []

hybrid_results = json_data['hybrid_results']
hybrid_fidelities = []

enhanced_results = json_data['enhanced_results']
enhanced_fidelities = []

def measured_x_calculator(results):
    if '01' in results.keys():
        return [results['01'], results.get('11', 0)]
    if '0 1' in results.keys():
        return [results['0 1'], results.get('1 1',0)]

for i in range(len(json_data['problem_list'])):
    ideal_x = QuantumLinearSystemSolver(problem=problem_list[i]).ideal_x_statevector.probabilities()

    measured_x = measured_x_calculator(canonical_results[i])
    measured_x /= np.linalg.norm(measured_x)
    canonical_fidelities.append(np.dot(ideal_x, measured_x))

    measured_x = measured_x_calculator(hybrid_results[i])
    measured_x /= np.linalg.norm(measured_x)
    hybrid_fidelities.append(np.dot(ideal_x, measured_x))

    measured_x = measured_x_calculator(enhanced_results[i])
    measured_x /= np.linalg.norm(measured_x)
    enhanced_fidelities.append(np.dot(ideal_x, measured_x))

errors = [np.sqrt(2*(1-fidelity)) for fidelity in canonical_fidelities]
print('ave can error : ',np.average(errors))

errors = [np.sqrt(2*(1-fidelity)) for fidelity in hybrid_fidelities]
print('ave hybrid error : ',np.average(errors))

errors = [np.sqrt(2*(1-fidelity)) for fidelity in enhanced_fidelities]
print('ave enhanced error : ',np.average(errors))

for key, value in json_data.items():
    if 'depth' in key:
        print(key, np.mean(value))