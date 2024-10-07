import os
from enhanced_hybrid_hhl import (HHL,
                                 ExampleQLSP,
                                 EnhancedHybridInversion,
                                 HybridInversion,
                                 CanonicalInversion,
                                 Lee_preprocessing,
                                 ExampleQLSP,
                                 )

import numpy as np
import json
from qiskit_aer import AerSimulator

# Import simulator for fidelity calculation
simulator = AerSimulator()

# Set file path and number of clock qubits
file_name = '0.json'
clock = 3

# choose values of lam. exclude 0 and 0.5
lam_list = list(np.linspace(0,0.5,100, endpoint=False))[1::]

#Create lists to store errors
can_results = []
hybrid_results = []
enhanced_results = []

for lam in lam_list:
    #Generate Problem
    problem = ExampleQLSP(lam=lam)

    #Canonical Fidelity
    Canonical_HHL = HHL(get_result_function='get_fidelity_result',                        
                        eigenvalue_inversion=CanonicalInversion,
                        )
    result = Canonical_HHL.estimate(problem=problem, 
                                    num_clock_qubits=clock,
                                    max_eigenvalue=1,
                                    )

    # Save canonical error
    fidelity = float(result.results_processed)
    error = np.sqrt(2*(1-fidelity))
    can_results.append(error)

    # Hybrid
    Hybrid_preprocessing=Lee_preprocessing(num_eval_qubits=clock, 
                                      backend = simulator, 
                                      max_eigenvalue=1,
                                      )
    
    Hybrid_HHL = HHL(get_result_function='get_fidelity_result',
                        pre_processing=Hybrid_preprocessing.estimate,
                        eigenvalue_inversion=HybridInversion,
                        )

    hybrid_result = Hybrid_HHL.estimate(problem=problem,                                              num_clock_qubits=clock,
                                              max_eigenvalue=1,
                                              )

    fidelity = float(hybrid_result.results_processed)
    error = np.sqrt(2*(1-fidelity))
    hybrid_results.append(error)
    
    # Enhanced
    Enhanced_preprocessing=Lee_preprocessing(num_eval_qubits=clock+2,
                                             backend = simulator,
                                             max_eigenvalue=1,
                                             )

    Enhanced_H_HHL = HHL(get_result_function='get_fidelity_result',
                        pre_processing=Enhanced_preprocessing.estimate,
                        eigenvalue_inversion=EnhancedHybridInversion,
                        )
    enhanced_result = Enhanced_H_HHL.estimate(problem=problem,
                                                num_clock_qubits=clock,
                                                max_eigenvalue=1,
                                                )
    
    fidelity = float(enhanced_result.results_processed)
    error = np.sqrt(2*(1-fidelity))
    enhanced_results.append(error)

data = {
    'lam' : lam_list,
    'Can.' : can_results,
    'Hybrid' : hybrid_results,
    'Enhanced' : enhanced_results,
}

script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)

with open(file_path, "w") as json_file:
   json.dump(data, json_file)

print("Data saved to", file_path)