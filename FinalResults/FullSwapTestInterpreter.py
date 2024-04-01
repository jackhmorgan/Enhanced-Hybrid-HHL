import numpy as np



def st_post_processing(result):
    if '0 1' in result.keys():
        counts_01 = result['0 1']
        if '1 1' in result.keys():
            counts_11 = result['1 1']
        else:
            counts_11 = 0

    else:
        counts_01 = result['1']
        counts_11 = result['3']
    if counts_01 <= counts_11:
        return 0
    else:
        prob_0 = counts_01/(counts_01+counts_11)
        return np.sqrt(2*prob_0 - 1)


import json
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
#file_name = 'simulator_N4_medgit ium_matrix_hhl.json'
#file_name = 'full_aria1_emulator_small_matrix_hhl_threshold4.json'
#file_name = 'benchmark_full_aria1_small_matrix_hhl.json'
file_name = 'aria1_simulator_to_emulator_small_matrix_hhl_threshold1.json'
# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

                    

canonical_results = json_data['canonical_results']
canonical_depths = json_data['canonical_depths']

preprocessing_depths = json_data['hybrid_preprocessing_depth']

hybrid_results = json_data['hybrid_results']
hybrid_depths = json_data['hybrid_depths']

enhanced_preprocessing_depths = json_data['enhacned_preprocessing_depth']

enhanced_results = json_data['enhanced_results']
enhanced_depths = json_data['enhanced_depths']

canonical_fidelities = []
hybrid_fidelities = []
enhanced_fidelities = []

for i, lam in enumerate(enhanced_results):
    #print('-----')
    #print('i = ', i)

    fidelity = st_post_processing(canonical_results[i])
    canonical_fidelities.append(fidelity)
    #print('canonical fidelity : ', fidelity)
    #print('canonical error : ', np.sqrt(2*(1-fidelity)))
    #print('canonical depth : ', canonical_depths[i])

    fidelity = st_post_processing(hybrid_results[i])
    hybrid_fidelities.append(fidelity)
    #print('hybrid fidelity : ', fidelity)
    #print('hybrid error : ', np.sqrt(2*(1-fidelity)))
    #print('hybrid depth : ', hybrid_depths[i])

    fidelity = st_post_processing(enhanced_results[i])
    enhanced_fidelities.append(fidelity)
    #print('enhanced fidelity : ', fidelity)
    #print('enhanced error : ', np.sqrt(2*(1-fidelity)))
    #print('enhanced depth : ', enhanced_depths[i])


fidelity = np.average(canonical_fidelities)
print('ave can error : ',np.sqrt(2*(1-fidelity)))
print('ave can depths : ',np.average(canonical_depths))

print('ave preprocessing depths : ', np.average(preprocessing_depths))

fidelity = np.average(hybrid_fidelities)
print('ave hybrid error : ',np.sqrt(2*(1-fidelity)))
print('ave hybrid depths : ',np.average(hybrid_depths))

print('ave enhanced preprocessing depths : ', np.average(enhanced_preprocessing_depths))

fidelity = np.average(enhanced_fidelities)
print('ave enhanced error : ',np.sqrt(2*(1-fidelity)))
print('ave enhanced depth : ',np.average(enhanced_depths))