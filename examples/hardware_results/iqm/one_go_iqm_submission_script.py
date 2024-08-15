import sys
import os

import json

from enhanced_hybrid_hhl import (HHL, 
                                 EnhancedHybridInversion,
                                 HybridInversion,
                                 Lee_preprocessing,
                                 list_preprocessing,
                                 QuantumLinearSystemProblem,
                                 ideal_preprocessing,
                                 ExampleQLSP,
                                 QuantumLinearSystemSolver,
                                 HHL_Result)

import numpy as np
from qiskit import transpile
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm import transpile_to_IQM

api_token = 'ZyWH0jUPqz+MKyKeuqCBp7GsJY1XaZH7VtjNQslUFaMGaW5hu+l/aIAAQFaKDS+D'
server_url = 'https://cocos.resonance.meetiqm.com/deneb'
backend = IQMProvider(server_url, token=api_token).get_backend()


shots = 1024
k_qubits = 3
l_qubits = k_qubits+2
probability_threshold = 2**(-k_qubits)

script_dir = os.path.dirname(os.path.realpath(__file__))

lam_list = [3/24, 4/24, 5/24, 6/24, 7/24, 8/24, 9/24, 10/24, 11/24]

problem_list = [ExampleQLSP(lam) for lam in lam_list]

used_problem_list = []
ideal_preprocessing_list = []

canonical_ids = []
canonical_depths = []
canonical_results = []

hybrid_preprocessing_list = []
hybrid_preprocessing_depth_list = []
hybrid_ids = []
hybrid_depths = []
hybrid_results = []

enhanced_preprocessing_list = []
enhanced_preprocessing_depth_list = []
enhanced_ids = []
enhanced_depths = []
enhanced_results = []

from qiskit import QuantumCircuit, ClassicalRegister

def get_iqm_result(circuit: QuantumCircuit,
                   problem: QuantumLinearSystemProblem,
                   shots: int = 1000,
                   ):
    hhl_circ = circuit
    c_reg = ClassicalRegister(2)
    hhl_circ.add_register(c_reg)

    # hhl_circ.add_register(q_reg)
    # hhl_circ.add_register(c_reg)

    # hhl_circ.prepare_state(statevector, q_reg[:-1])
    # hhl_circ.append(st, list(range(-st.num_qubits,0)))
    # hhl_circ.measure(0,0)
    hhl_circ.measure(-1,c_reg[1])

    circuit = transpile_to_IQM(hhl_circ, backend=backend)
    result = HHL_Result()
    result.depth = circuit.count_ops()

    job = backend.run(circuit, shots=shots)
    result.circuit_results = job
    return result

def get_iqm_result_preprocessing(circ):
    transp = transpile_to_IQM(circ, backend)
    job = backend.run(transp, shots=1000)
    dictionary = job.result().get_counts()
    result_dict = {int(key, 2): value for key, value in dictionary.items()}
    return result_dict

def get_iqm_result_preprocessing_transpiler(circ):
    transp = transpile_to_IQM(circ, backend)
    #job = backend.run(transp, shots=1024)
    return transp.count_ops()

for iteration in range(2,5):
    for problem in problem_list:

        used_problem_list.append(problem.A_matrix.tolist())

        solution= QuantumLinearSystemSolver(problem)
        ideal_x = solution.ideal_x_statevector
        ideal_preprocessing_list.append(ideal_preprocessing(problem))

        '''Canonical'''

        #Canonical_HHL = HHL(get_result_function='get_ionq_result',
        #                    eigenvalue_inversion=CanonicalInversion,
        #                    backend=backend,
        #                    noise_model=noise_model,
        #                    shots=shots,
        #                    statevector = ideal_x)
        #canonical_result = Canonical_HHL.estimate(problem=problem,
        #                                          num_clock_qubits=k_qubits,
        #                                          max_eigenvalue=1,
        #                                          quantum_conditional_logic=False,
        #                                         )
        #canonical_ids.append(canonical_result.circuit_results.job_id())
        #canonical_results.append(canonical_result.circuit_results.get_counts())
        #canonical_depths.append(canonical_result.results_processed)

        '''Hybrid'''
        h_preprocessing=Lee_preprocessing(num_eval_qubits=k_qubits, 
                                           max_eigenvalue=1, 
                                           get_result_function=get_iqm_result_preprocessing_transpiler,
                                           wait_for_result=False   
                                          )
        hybrid_preprocessing_depth_list.append(h_preprocessing.estimate(problem))

        h_preprocessing=Lee_preprocessing(num_eval_qubits=k_qubits, 
                                           max_eigenvalue=1, 
                                           get_result_function=get_iqm_result_preprocessing,
                                           wait_for_result=True   
                                          )
        eigenvalue_list, eigenbasis_projection_list = h_preprocessing.estimate(problem)
        hybrid_preprocessing_list.append((eigenvalue_list, eigenbasis_projection_list))

        hhl_preprocessing = list_preprocessing(eigenvalue_list, eigenbasis_projection_list)
        #hhl_preprocessing = ideal_preprocessing

        H_HHL = HHL(preprocessing=hhl_preprocessing,
                    eigenvalue_inversion=HybridInversion,
                    )
        hybrid_result = H_HHL.estimate(problem=problem,
                                       num_clock_qubits=k_qubits,
                                       max_eigenvalue=1,
                                       quantum_conditional_logic=False,
                                       get_result_function=get_iqm_result,
                                       )
                                      
        hybrid_ids.append(hybrid_result.circuit_results.job_id())
        hybrid_results.append(hybrid_result.circuit_results.result().get_counts())
        hybrid_depths.append(hybrid_result.results_processed)


        '''Enhanced'''
        e_preprocessing=Lee_preprocessing(num_eval_qubits=l_qubits, 
                                           max_eigenvalue=1, 
                                           get_result_function=get_iqm_result_preprocessing_transpiler,
                                           wait_for_result=False   
                                          )
        
        enhanced_preprocessing_depth_list.append(e_preprocessing.estimate(problem))

        e_preprocessing=Lee_preprocessing(num_eval_qubits=l_qubits,
                                          max_eigenvalue=1,
                                          get_result_function=get_iqm_result_preprocessing
                                         )

        enhanced_eigenvalue_list, enhanced_eigenbasis_projection_list = e_preprocessing.estimate(problem)
        enhanced_preprocessing_list.append((enhanced_eigenvalue_list, enhanced_eigenbasis_projection_list))

        hhl_preprocessing = list_preprocessing(enhanced_eigenvalue_list, enhanced_eigenbasis_projection_list)
        #hhl_preprocessing = ideal_preprocessing

        Enhanced_H_HHL = HHL(preprocessing=hhl_preprocessing,
                            eigenvalue_inversion=EnhancedHybridInversion,
                            )
        enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                                    num_clock_qubits=k_qubits,
                                                    max_eigenvalue=1,
                                                    quantum_conditional_logic=False,
                                                    get_result_function=get_iqm_result,
                                                    )
        enhanced_ids.append(enhanced_result.circuit_results.job_id())
        enhanced_results.append(enhanced_result.circuit_results.result().get_counts())
        enhanced_depths.append(enhanced_result.results_processed)



    data = {
        'problem_list' : used_problem_list,
        'shots' : shots,
        'backend' : 'Deneb',
        'preprocessing_backend' : 'Deneb',
        'ideal_preprocessing_list' : ideal_preprocessing_list,
        'probability_threshold' : probability_threshold,

        'canonical_ids' : canonical_ids,
        'canonical_depths' : canonical_depths,
        'canonical_results' : canonical_results,

        'hybrid_preprocessing_list' : hybrid_preprocessing_list,
        'hybrid_preprocessing_depth' : hybrid_preprocessing_depth_list,

        'hybrid_ids' : hybrid_ids,
        'hybrid_depths' : hybrid_depths,
        'hybrid_results' : hybrid_results,

        'enhanced_preprocessing_list' : enhanced_preprocessing_list,
        'enhacned_preprocessing_depth' : enhanced_preprocessing_depth_list,

        'enhanced_ids' : enhanced_ids,
        'enhanced_depths' : enhanced_depths,
        'enhanced_results' : enhanced_results,
    }

    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = 'iqm_deneb_small_matrix_hhl'+str(iteration)+'.json'
    # Define the file path
    file_path = os.path.join(script_dir, file_name)

    with open(file_path, "w") as json_file:
        json.dump(data, json_file)