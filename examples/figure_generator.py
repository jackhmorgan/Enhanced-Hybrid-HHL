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
from qiskit_ibm_runtime import QiskitRuntimeService, Session

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-ncsu/nc-state/amplitude-estima',
    token='35e7316e04600dcc5e1b36a25b8eb9c61601d1456fe4d0ddb1604a8ef1847f0614516de700c3f5d9bac07141ade760115a07f1a706307d04fd16066d1a0dd109'
)

backend = service.get_backend('ibm_torino')

simulator = AerSimulator()

file_name = 'example_figuredepth.json'

clock = 3

lam_list = list(np.linspace(0,0.5,3, endpoint=False))[1::]
cann_results = []
hybrid_results = []
enhanced_results = []
fixed_enhanced_results = []

for lam in lam_list:
    problem = ExampleQLSP(lam=lam)

    Cannonical_HHL = HHL(get_result_function='get_circuit_depth_result',
                        eigenvalue_inversion=CannonicalInversion,
                        backend = backend
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
    
    Yalovetsky_H_HHL = HHL(get_result_function='get_circuit_depth_result',
                        pre_processing=y_preprocessing.estimate,
                        eigenvalue_inversion=HybridInversion,
                        backend=backend
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

    Enhanced_H_HHL = HHL(get_result_function='get_circuit_depth_result',
                        pre_processing=e_preprocessing.estimate,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        backend=backend
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

    fixed_Enhanced_H_HHL = HHL(get_result_function='get_circuit_depth_result',
                        pre_processing=e_preprocessing,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        backend=backend,
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