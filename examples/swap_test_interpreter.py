from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

service = QiskitRuntimeService()



def st_post_processing(job_id):
    result = service.job(job_id=job_id).result().quasi_dists[0]
    counts_01 = 0
    counts_11 = 1
    if not result==None:
        counts_01 = result[1]
        counts_11 = result[3]
    if counts_01 <= counts_11:
        return 0
    else:
        prob_0 = counts_01/(counts_01+counts_11)
        return np.sqrt(2*prob_0 - 1)


import json
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'simulator_medium_matrix_hhl.json'

# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

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
lam_list = json_data['_list']
canonical_ids = json_data['canonical_ids']
canonical_depths = json_data['canonical_depths']
hybrid_ids = json_data['hybrid_ids']
hybrid_depths = json_data['hybrid_depths']
enhanced_ids = json_data['enhanced_ids']
enhanced_depths = json_data['enhanced_depths']

canonical_fidelities = []
hybrid_fidelities = []
enhanced_fidelities = []

for i, lam in enumerate(lam_list):
    print('lam : ', i)
    fidelity = st_post_processing(job_id=canonical_ids[i])
    canonical_fidelities.append(fidelity)
    print('-------')
    print('canonical')
    print('fidelity : ', fidelity)
    print('error : ', np.sqrt(2*(1-fidelity)))
    print('depth : ', canonical_depths[i])
    print('-------')
    fidelity = st_post_processing(job_id=hybrid_ids[i])
    hybrid_fidelities.append(fidelity)
    print('hybrid')
    print('fidelity : ', fidelity)
    print('error : ', np.sqrt(2*(1-fidelity)))
    print('depth : ', canonical_depths[i])
    print('-------')
    fidelity = st_post_processing(job_id=enhanced_ids[i])
    enhanced_fidelities.append(fidelity)
    print('enhanced')
    print('fidelity : ', fidelity)
    print('error : ', np.sqrt(2*(1-fidelity)))
    print('depth : ', canonical_depths[i])


fidelity = np.average(canonical_fidelities)
print(np.sqrt(2*(1-fidelity)))
print(np.average(canonical_depths))
fidelty = np.average(hybrid_fidelities)
print(np.sqrt(2*(1-fidelity)))
print(np.average(hybrid_depths))
fidelity = np.average(enhanced_fidelities)
print(np.sqrt(2*(1-fidelity)))
print(np.average(enhanced_depths))
fid_array = np.array(e-c for e, c in zip(enhanced_fidelities, canonical_fidelities))

#print(np.where(fid_array == fid_array.max()))
print('---------')
for can, enhan in zip(canonical_fidelities, enhanced_fidelities):
    c =np.sqrt(2*(1-can))
    e = np.sqrt(2*(1-enhan))
    print(c-e)