import sys
import os

import json

import numpy as np
from enhanced_hybrid_hhl import (HHL, 
                                 Lee_preprocessing,  
                                 HybridInversion, 
                                 QuantumLinearSystemProblem, 
                                 QuantumLinearSystemSolver,
                                 EnhancedHybridInversion,
                                 HHL_Result,
                                 ExampleQLSP,
                                 ideal_preprocessing,
                                 CanonicalInversion,
                                 list_preprocessing)

import numpy as np

from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister

from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm import transpile_to_IQM
from qiskit.providers.fake_provider import FakeHanoi

script_dir = os.path.dirname(os.path.realpath(__file__))
preprocessing_file_name = 'simulator_small_matrix_preprocessing.json'

# Define the file path
file_path = os.path.join(script_dir, preprocessing_file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

server_url = 'https://cocos.resonance.meetiqm.com/deneb'
backend = IQMProvider(server_url, token=api_token).get_backend()

backend = FakeHanoi()

lam_list = json_data['lam_list']
problem_list = [ExampleQLSP(lam) for lam in lam_list]
fixed_result_list = json_data['fixed']
enhanced_fixed_result_list = json_data['enhanced_fixed']


k_qubits = 3
probability_threshold = 0 #2**(-k_qubits)

ideal_preprocessing_list = []
used_problem_list = []

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

    #circuit = transpile_to_IQM(hhl_circ, backend=backend)
    circuit = transpile(hhl_circ, backend=backend)
    result = HHL_Result()
    result.depth = circuit.count_ops()
    result.circuit_results = circuit.depth()
    return result

def get_iqm_result_preprocessing_transpiler(circ):
    #transp = transpile_to_IQM(circ, backend)
    transp = transpile(circ, backend=backend)
    #job = backend.run(transp, shots=1024)
    return transp.depth()

for iteration in range(3,6):
    for i, problem in enumerate(problem_list):    
        used_problem_list.append(problem.A_matrix.tolist())
        solution= QuantumLinearSystemSolver(problem)
        ideal_x = solution.ideal_x_statevector
        
        ideal_preprocessing_list.append(ideal_preprocessing(problem))

        Cannonical_HHL = HHL(eigenvalue_inversion=CanonicalInversion,
                            )

        canonical_result = Cannonical_HHL.estimate(problem=problem, 
                                        num_clock_qubits=k_qubits,
                                        max_eigenvalue=1,
                                        quantum_conditional_logic=False,
                                        get_result_function=get_iqm_result,
                                        )

        
        canonical_results.append(canonical_result.circuit_results)
        canonical_depths.append(canonical_result.depth)

        h_preprocessing=Lee_preprocessing(num_eval_qubits=k_qubits, 
                                           max_eigenvalue=1, 
                                           get_result_function=get_iqm_result_preprocessing_transpiler,
                                           wait_for_result=False   
                                          )
        hybrid_preprocessing_depth_list.append(h_preprocessing.estimate(problem))


        y_preprocessing = list_preprocessing(fixed_result_list[i][0], fixed_result_list[i][1])
        Yalovetsky_H_HHL = HHL(preprocessing=y_preprocessing,
                            eigenvalue_inversion=HybridInversion,
                            )
        hybrid_result = Yalovetsky_H_HHL.estimate(problem=problem,
                                                    num_clock_qubits=k_qubits,
                                                    max_eigenvalue=1,
                                                    quantum_conditional_logic=False,
                                        get_result_function=get_iqm_result,
                                        )

        #hybrid_ids.append(hybrid_result.circuit_results.job_id())
        hybrid_results.append(hybrid_result.circuit_results)
        hybrid_depths.append(hybrid_result.depth)

        e_preprocessing=Lee_preprocessing(num_eval_qubits=k_qubits+2, 
                                           max_eigenvalue=1, 
                                           get_result_function=get_iqm_result_preprocessing_transpiler,
                                           wait_for_result=False   
                                          )
        
        enhanced_preprocessing_depth_list.append(e_preprocessing.estimate(problem))
        
        e_preprocessing = list_preprocessing(enhanced_fixed_result_list[i][0], enhanced_fixed_result_list[i][1])
        
        Enhanced_H_HHL = HHL(preprocessing=e_preprocessing,
                            eigenvalue_inversion=EnhancedHybridInversion,
                            )
        enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                                    num_clock_qubits=k_qubits,
                                                    max_eigenvalue=1,
                                                    quantum_conditional_logic=False,
                                                    probability_threshold=probability_threshold,
                                                    get_result_function=get_iqm_result,
                                                    )
        #enhanced_ids.append(enhanced_result.circuit_results.job_id())
        enhanced_results.append(enhanced_result.circuit_results)
        enhanced_depths.append(enhanced_result.depth)
            
            

    data = {
        'problem_list' : used_problem_list,
        'shots' : 1000,
        'backend' : backend.name(),
        'preprocessing_backend' : preprocessing_file_name,
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
    file_name = 'hanoi_N2_depth_matrix_hhl'+str(iteration)+'.json'
    # Define the file path
    file_path = os.path.join(script_dir, file_name)

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)