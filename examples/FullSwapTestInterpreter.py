from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

service = QiskitRuntimeService()



def st_post_processing(result):
    counts_01 = result['0 1']
    counts_11 = result['1 1']
    if counts_01 <= counts_11:
        return 0
    else:
        prob_0 = counts_01/(counts_01+counts_11)
        return np.sqrt(2*prob_0 - 1)


import json
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'full_ionq_aria_small_matrix_hhl.json'
#file_name = 'benchmark_full_aria1_small_matrix_hhl.json'
#file_name = 'ionq_aria_small_matrix_hhl.json'
# Define the file path


lam_list =  [[-23/24, 11/24],
                             [-22/24, 11/24],
                             [-21/24, 11/24],
                             [-20/24, 11/24],
                             [-19/24, 11/24],
                             [-18/24, 11/24],
                             [-19/24, 4/24],
                             [-19/24, 5/24],
                             [-19/24, 6/24],
                             [-19/24, 7/24],
                             [-19/24, 9/24],                             
                             [-19/24, 10/24]]

canonical_results = json_data['canonical_results']
canonical_depths = json_data['canonical_depths']

hybrid_results = json_data['hybrid_results']
hybrid_depths = json_data['hybrid_depths']

enhanced_results = json_data['enhanced_results']
enhanced_depths = json_data['enhanced_depths']

canonical_fidelities = []
hybrid_fidelities = []
enhanced_fidelities = []

for i, lam in enumerate(enhanced_results):
    print('-----')
    print('i = ', i)

    fidelity = st_post_processing(canonical_results[i])
    canonical_fidelities.append(fidelity)
    print('canonical fidelity : ', fidelity)
    print('canonical error : ', np.sqrt(2*(1-fidelity)))
    print('canonical depth : ', canonical_depths[i])

    fidelity = st_post_processing(hybrid_results[i])
    hybrid_fidelities.append(fidelity)
    print('hybrid fidelity : ', fidelity)
    print('hybrid error : ', np.sqrt(2*(1-fidelity)))
    print('hybrid depth : ', hybrid_depths[i])

    fidelity = st_post_processing(enhanced_results[i])
    enhanced_fidelities.append(fidelity)
    print('enhanced fidelity : ', fidelity)
    print('enhanced error : ', np.sqrt(2*(1-fidelity)))
    print('enhanced depth : ', enhanced_depths[i])


fidelity = np.average(canonical_fidelities)
print('ave can error : ',np.sqrt(2*(1-fidelity)))
print('ave can depths : ',np.average(canonical_depths))

fidelity = np.average(hybrid_fidelities)
print('ave hybrid error : ',np.sqrt(2*(1-fidelity)))
print('ave hybrid depths : ',np.average(hybrid_depths))

fidelity = np.average(enhanced_fidelities)
print('ave enhanced error : ',np.sqrt(2*(1-fidelity)))
print('ave enhanced depth : ',np.average(enhanced_depths))