import json
import os
import numpy as np

from qiskit_ionq import IonQProvider

provider = IonQProvider()
backend = provider.get_backend("ionq_simulator")

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

script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'ideal_to_simulator_N4_matrix_hhl6.json'

# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)


enhanced_ids = json_data['enhanced_ids']
enhanced_depths = json_data['enhanced_depths']

enhanced_results = []
enhanced_fidelities = []

#hybrid_results = [{'0 0': 595, '0 1': 183, '1 0': 192, '1 1': 54}, {'0 0': 509, '0 1': 273, '1 0': 167, '1 1': 76}, {'0 0': 435, '0 1': 343, '1 0': 163, '1 1': 84}, {'0 1': 403, '0 0': 378, '1 0': 173, '1 1': 71}, {'0 1': 437, '0 0': 352, '1 0': 185, '1 1': 51}, {'0 1': 486, '0 0': 321, '1 0': 151, '1 1': 67}, {'0 1': 450, '0 0': 347, '1 0': 155, '1 1': 73}, {'0 1': 445, '0 0': 404, '1 0': 110, '1 1': 66}, {'0 1': 440, '0 0': 495, '1 1': 41, '1 0': 49}]
#enhanced_results = [{'0 0': 561, '1 0': 238, '0 1': 175, '1 1': 50}, {'0 0': 514, '0 1': 246, '1 0': 198, '1 1': 66}, {'0 0': 475, '0 1': 297, '1 0': 181, '1 1': 72}, {'0 1': 441, '0 0': 345, '1 0': 174, '1 1': 65}, {'0 1': 500, '0 0': 311, '1 0': 152, '1 1': 62}, {'0 1': 572, '0 0': 266, '1 0': 123, '1 1': 63}, {'0 1': 570, '0 0': 269, '1 0': 115, '1 1': 70}, {'0 1': 595, '0 0': 292, '1 1': 69, '1 0': 68}, {'0 1': 641, '0 0': 273, '1 1': 72, '1 0': 38}]

for enhanced_id in enhanced_ids:

    #
    enhanced_result = backend.retrieve_job(enhanced_id).get_counts()
    enhanced_results.append(enhanced_result)
    #print('hybrid fidelity : ', fidelity)
    #print('hybrid error : ', np.sqrt(2*(1-fidelity)))
    #print('hybrid depth : ', hybrid_depths[i])

    fidelity = st_post_processing(enhanced_result)
    enhanced_fidelities.append(fidelity)
    #print('enhanced fidelity : ', fidelity)
    #print('enhanced error : ', np.sqrt(2*(1-fidelity)))
    #print('enhanced depth : ', enhanced_depths[i])


errors = [np.sqrt(2*(1-fidelity)) for fidelity in enhanced_fidelities]
print('ave enhanced fidelity :', np.average(enhanced_fidelities))
print('ave enhanced error : ',np.average(errors))
print('ave enhanced depth : ',np.average(enhanced_depths))

json_data['enhanced_results']=enhanced_results

with open(file_path, 'w') as file:
    json.dump(json_data, file)