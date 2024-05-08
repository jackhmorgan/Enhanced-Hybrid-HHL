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
file_name = 'simulator_to_aria1_N2_matrix_hhl.json'
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

#hybrid_results = [{'0 0': 595, '0 1': 183, '1 0': 192, '1 1': 54}, {'0 0': 509, '0 1': 273, '1 0': 167, '1 1': 76}, {'0 0': 435, '0 1': 343, '1 0': 163, '1 1': 84}, {'0 1': 403, '0 0': 378, '1 0': 173, '1 1': 71}, {'0 1': 437, '0 0': 352, '1 0': 185, '1 1': 51}, {'0 1': 486, '0 0': 321, '1 0': 151, '1 1': 67}, {'0 1': 450, '0 0': 347, '1 0': 155, '1 1': 73}, {'0 1': 445, '0 0': 404, '1 0': 110, '1 1': 66}, {'0 1': 440, '0 0': 495, '1 1': 41, '1 0': 49}]
#enhanced_results = [{'0 0': 561, '1 0': 238, '0 1': 175, '1 1': 50}, {'0 0': 514, '0 1': 246, '1 0': 198, '1 1': 66}, {'0 0': 475, '0 1': 297, '1 0': 181, '1 1': 72}, {'0 1': 441, '0 0': 345, '1 0': 174, '1 1': 65}, {'0 1': 500, '0 0': 311, '1 0': 152, '1 1': 62}, {'0 1': 572, '0 0': 266, '1 0': 123, '1 1': 63}, {'0 1': 570, '0 0': 269, '1 0': 115, '1 1': 70}, {'0 1': 595, '0 0': 292, '1 1': 69, '1 0': 68}, {'0 1': 641, '0 0': 273, '1 1': 72, '1 0': 38}]

for i, lam in enumerate(enhanced_results):
    #print('-----')
    #print('i = ', i)

    #fidelity = st_post_processing(canonical_results[i])
    #canonical_fidelities.append(fidelity)
    #print('canonical fidelity : ', fidelity)
    #print('canonical error : ', np.sqrt(2*(1-fidelity)))
    #print('canonical depth : ', canonical_depths[i])

    fidelity = st_post_processing(hybrid_results[i])
    #
    hybrid_fidelities.append(fidelity)
    #print('hybrid fidelity : ', fidelity)
    #print('hybrid error : ', np.sqrt(2*(1-fidelity)))
    #print('hybrid depth : ', hybrid_depths[i])

    fidelity = st_post_processing(enhanced_results[i])
    enhanced_fidelities.append(fidelity)
    #print('enhanced fidelity : ', fidelity)
    #print('enhanced error : ', np.sqrt(2*(1-fidelity)))
    #print('enhanced depth : ', enhanced_depths[i])

errors = [np.sqrt(2*(1-fidelity)) for fidelity in canonical_fidelities]
print('ave can error : ',np.average(errors))
#print('ave can depths : ',np.average(canonical_depths))

#print('ave preprocessing depths : ', np.average(preprocessing_depths))

errors = [np.sqrt(2*(1-fidelity)) for fidelity in hybrid_fidelities]
print('ave hybrid error : ',np.average(errors))
#print('ave hybrid depths : ',np.average(hybrid_depths))

#print('ave enhanced preprocessing depths : ', np.average(enhanced_preprocessing_depths))

errors = [np.sqrt(2*(1-fidelity)) for fidelity in enhanced_fidelities]
print('ave enhanced error : ',np.average(errors))
#print('ave enhanced depth : ',np.average(enhanced_depths))