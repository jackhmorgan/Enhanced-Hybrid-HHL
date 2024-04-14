import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')
import json
import os

from enhanced_hybrid_hhl import (HHL, 
                                 EnhancedHybridInversion,
                                 HybridInversion,
                                 CannonicalInversion,
                                 Lee_preprocessing,
                                 Yalovetsky_preprocessing,
                                 ideal_preprocessing,
                                 QuantumLinearSystemProblem,
                                 ExampleQLSP,
                                 QuantumLinearSystemSolver,
                                 list_preprocessing)
import numpy as np
from qiskit.quantum_info import random_hermitian
from qiskit.circuit.library import StatePreparation, HamiltonianGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ionq.ionq_provider import IonQProvider

ionq_provider = IonQProvider(token='eqMVRwVhZkVIX9lEBlIMYhxFtneQnCD5')

backend = ionq_provider.get_backend('ionq_simulator.aria')

script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'torino_small_matrix_preprocessing.json'

# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

#backend = service.get_backend('ibm_torino')
#backend = service.get_backend('ibmq_qasm_simulator')

problem_list = json_data['lam_list']
fixed_result_list = json_data['fixed']
enhanced_fixed_result_list = json_data['enhanced_fixed']


k_qubits = 3

canonical_ids = []
hybrid_ids = []
enhanced_ids = []

canonical_depths = []
hybrid_depths = []
enhanced_depths = []


for i, json_problem in enumerate(problem_list):
    #A_matrix = json_problem['A_matrix']
    #b_vector = json_problem['b_vector']

    #problem = QuantumLinearSystemProblem(A_matrix=A_matrix, b_vector=b_vector)

    problem = ExampleQLSP(json_problem)

    solution= QuantumLinearSystemSolver(problem)
    ideal_x = solution.ideal_x_statevector

    Cannonical_HHL = HHL(get_result_function='get_ionq_result',
                        eigenvalue_inversion=CannonicalInversion,
                        backend=backend,
                        statevector = ideal_x)

    canonical_result = Cannonical_HHL.estimate(problem=problem, 
                                    num_clock_qubits=k_qubits,
                                    max_eigenvalue=1,
                                    quantum_conditional_logic=False,
                                    )

    canonical_ids.append(canonical_result.circuit_results.job_id())
    canonical_depths.append(canonical_result.results_processed)


    y_preprocessing = list_preprocessing(fixed_result_list[i][0], fixed_result_list[i][1])
    Yalovetsky_H_HHL = HHL(get_result_function='get_ionq_result',
                        pre_processing=y_preprocessing,
                        eigenvalue_inversion=HybridInversion,
                        backend=backend,
                        statevector = ideal_x)

    hybrid_result = Yalovetsky_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=k_qubits,
                                                max_eigenvalue=1,
                                                quantum_conditional_logic=False,
                                                )

    hybrid_ids.append(hybrid_result.circuit_results.job_id())
    hybrid_depths.append(hybrid_result.results_processed)

    e_preprocessing = list_preprocessing(enhanced_fixed_result_list[i][0], enhanced_fixed_result_list[i][1])

    Enhanced_H_HHL = HHL(get_result_function='get_ionq_result',
                        pre_processing=e_preprocessing,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        backend=backend,
                        statevector = ideal_x)
    enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=k_qubits,
                                                max_eigenvalue=1,
                                                quantum_conditional_logic=False,
                                                )
    enhanced_ids.append(enhanced_result.circuit_results.job_id())
    enhanced_depths.append(enhanced_result.results_processed)
        
        

data = {
    'problem_list' : problem_list,
    'canonical_ids' : canonical_ids,
    'canonical_depths' : canonical_depths,
    'hybrid_ids' : hybrid_ids,
    'hybrid_depths' : hybrid_depths,
    'enhanced_ids' : enhanced_ids,
    'enhanced_depths' : enhanced_depths
}

script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'torino_to_ionq_small_matrix_hhl.json'
# Define the file path
file_path = os.path.join(script_dir, file_name)

with open(file_path, "w") as json_file:
    json.dump(data, json_file)