import json
import os
import numpy as np

from enhanced_hybrid_hhl import QuantumLinearSystemSolver, QuantumLinearSystemProblem

file_name = 'aria1_emulator_N2_matrix_hhl0.json'
#file_name = 'deneb_N2_depth_matrix_hhl3.json'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

problem_list = [QuantumLinearSystemProblem(A_matrix=problem, b_vector=[1,0]) for problem in json_data['problem_list']]

canonical_depths = json_data['canonical_results']
canonical_fidelities = []

hybrid_depths = json_data['hybrid_results']
hybrid_fidelities = []

enhanced_depths = json_data['enhanced_results']
enhanced_fidelities = []

enhanced_preprocessing_depths = json_data['enhacned_preprocessing_depth']
hybrid_preprocessing_depths = json_data['hybrid_preprocessing_depth']

list_dict = {'canonical_depths': canonical_depths,
             'hybrid_depths': hybrid_depths,
             'enhanced_depths': enhanced_depths,
             'hybrid_preprocessing_depths': hybrid_preprocessing_depths,
             'enhanced_preprocessing_depths' : enhanced_preprocessing_depths}

print([(key, np.mean(value)) for key, value in list_dict.items()])
