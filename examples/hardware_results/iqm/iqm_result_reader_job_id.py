import json
import os
import numpy as np

from enhanced_hybrid_hhl import QuantumLinearSystemSolver, QuantumLinearSystemProblem

from iqm.qiskit_iqm import IQMProvider

server_url = 'https://cocos.resonance.meetiqm.com/deneb'
backend = IQMProvider(server_url).get_backend()
for iteration in [0,1,3,4,5]:
    file_name = 'simulator_to_deneb_N2_matrix_hhl'+str(iteration)+'.json'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define the file path
    file_path = os.path.join(script_dir, file_name)
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    problem_list = [QuantumLinearSystemProblem(A_matrix=problem, b_vector=[1,0]) for problem in json_data['problem_list']]

    canonical_ids = json_data['canonical_ids']
    canonical_results = []
    canonical_fidelities = []

    hybrid_ids = json_data['hybrid_ids']
    hybrid_results = []
    hybrid_fidelities = []

    enhanced_ids = json_data['enhanced_ids']
    enhanced_fidelities = []
    enhanced_results = []


    for i in range(len(json_data['problem_list'])):
        ideal_x = QuantumLinearSystemSolver(problem=problem_list[i]).ideal_x_statevector.probabilities()

        json_data['canonical_results'].append(backend.retrieve_job(canonical_ids[i]).result().get_counts())
        json_data['hybrid_results'].append(backend.retrieve_job(hybrid_ids[i]).result().get_counts())
        json_data['enhanced_results'].append(backend.retrieve_job(enhanced_ids[i]).result().get_counts())

        # measured_x = [canonical_results[i]['01'], canonical_results[i]['11']]
        # measured_x /= np.linalg.norm(measured_x)
        # canonical_fidelities.append(np.dot(ideal_x, measured_x))

        # measured_x = [hybrid_results[i]['01'], hybrid_results[i]['11']]
        # measured_x /= np.linalg.norm(measured_x)
        # hybrid_fidelities.append(np.dot(ideal_x, measured_x))

        # measured_x = [enhanced_results[i]['01'], enhanced_results[i]['11']]
        # measured_x /= np.linalg.norm(measured_x)
        # enhanced_fidelities.append(np.dot(ideal_x, measured_x))

    # errors = [np.sqrt(2*(1-fidelity)) for fidelity in canonical_fidelities]
    # print('ave can error : ',np.average(errors))

    # errors = [np.sqrt(2*(1-fidelity)) for fidelity in hybrid_fidelities]
    # print('ave hybrid error : ',np.average(errors))

    # errors = [np.sqrt(2*(1-fidelity)) for fidelity in enhanced_fidelities]
    # print('ave enhanced error : ',np.average(errors))
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)