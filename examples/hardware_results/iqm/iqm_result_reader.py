import json
import os
import numpy as np

from enhanced_hybrid_hhl import QuantumLinearSystemSolver, QuantumLinearSystemProblem

file_name = 'iqm_deneb_small_matrix_hhl3.json'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

problem_list = [QuantumLinearSystemProblem(A_matrix=problem, b_vector=[1,0]) for problem in json_data['problem_list']]

hybrid_results = json_data['hybrid_results']
hybrid_fidelities = []

enhanced_results = json_data['enhanced_results']
enhanced_fidelities = []


for i in range(len(json_data['problem_list'])):
    ideal_x = QuantumLinearSystemSolver(problem=problem_list[i]).ideal_x_statevector.probabilities()

    measured_x = [hybrid_results[i]['01'], hybrid_results[i]['11']]
    measured_x /= np.linalg.norm(measured_x)
    hybrid_fidelities.append(np.dot(ideal_x, measured_x))

    measured_x = [enhanced_results[i]['01'], enhanced_results[i]['11']]
    measured_x /= np.linalg.norm(measured_x)
    enhanced_fidelities.append(np.dot(ideal_x, measured_x))


errors = [np.sqrt(2*(1-fidelity)) for fidelity in hybrid_fidelities]
print('ave hybrid error : ',np.average(errors))

errors = [np.sqrt(2*(1-fidelity)) for fidelity in enhanced_fidelities]
print('ave hybrid error : ',np.average(errors))
