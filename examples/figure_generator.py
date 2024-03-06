import sys
import os
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')
from enhanced_hybrid_hhl import (HHL,
                                 ExampleQLSP,
                                 EnhancedHybridInversion,
                                 HybridInversion,
                                 CannonicalInversion,
                                 Lee_preprocessing,
                                 Yalovetsky_preprocessing,
                                 ideal_preprocessing,
                                 QuantumLinearSystemProblem,
                                 ExampleQLSP,
                                 QuantumLinearSystemSolver)

import numpy as np
import json
from qiskit_aer import AerSimulator

simulator = AerSimulator()

file_name = 'example_figure100notfixed.json'

clock = 3

lam_list = list(np.linspace(0,0.5,100, endpoint=False))[1::]
cann_results = []
hybrid_results = []
enhanced_results = []
fixed_enhanced_results = []

for lam in lam_list:
    problem = ExampleQLSP(lam=lam)

    Cannonical_HHL = HHL(get_result_function='get_fidelity_result',
                        eigenvalue_inversion=CannonicalInversion,
                        )

    result = Cannonical_HHL.estimate(problem=problem, 
                                    num_clock_qubits=clock,
                                    max_eigenvalue=1,
                                    )

    fidelity = float(result.results_processed)
    error = np.sqrt(2*(1-fidelity))
    cann_results.append(error)

    #y_preprocessing=Lee_preprocessing(num_eval_qubits=clock, 
    #                                  backend = simulator, 
    #                                max_eigenvalue=1)

    y_preprocessing=Yalovetsky_preprocessing(clock=clock,
                                             backend = simulator)
    
    Yalovetsky_H_HHL = HHL(get_result_function='get_fidelity_result',
                        pre_processing=y_preprocessing.estimate,
                        eigenvalue_inversion=HybridInversion,
                        )

    hybrid_result = Yalovetsky_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=clock,
                                                #max_eigenvalue=1,
                                                )

    fidelity = float(hybrid_result.results_processed)
    error = np.sqrt(2*(1-fidelity))
    hybrid_results.append(error)


    #e_preprocessing=Lee_preprocessing(num_eval_qubits=clock+2, 
    #                                backend=simulator, 
    #                                max_eigenvalue=1)
    
    e_preprocessing=Yalovetsky_preprocessing(clock=clock+2,
                                            backend = simulator)

    Enhanced_H_HHL = HHL(get_result_function='get_fidelity_result',
                        pre_processing=e_preprocessing.estimate,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        )
    enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=clock,
                                                #max_eigenvalue=1,
                                                )
    
    fidelity = float(enhanced_result.results_processed)
    error = np.sqrt(2*(1-fidelity))
    enhanced_results.append(error)

    e_preprocessing=Lee_preprocessing(num_eval_qubits=clock+2, 
                                    backend=simulator, 
                                    max_eigenvalue=1)

    fixed_Enhanced_H_HHL = HHL(get_result_function='get_fidelity_result',
                        pre_processing=e_preprocessing,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        )
    
    fixed_enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=clock,
                                                max_eigenvalue=1,
                                                )
    
    fidelity = float(fixed_enhanced_result.results_processed)
    error = np.sqrt(2*(1-fidelity))
    fixed_enhanced_results.append(error)
data = {
    'lam' : lam_list,
    'Cann.' : cann_results,
    'Hybrid' : hybrid_results,
    'Enhanced' : enhanced_results,
}




script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)

with open(file_path, "w") as json_file:
   json.dump(data, json_file)

print("Data saved to", file_path)