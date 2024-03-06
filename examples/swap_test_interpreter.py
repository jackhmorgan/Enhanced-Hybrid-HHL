from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

service = QiskitRuntimeService()



def st_post_processing(job_id):
    result = service.job(job_id=job_id).result().quasi_dists[0]
    if not result==None:
        counts_01 = result[1]
        counts_11 = result[3]
    prob_0 = counts_01/(counts_01+counts_11)
    return np.sqrt(2*prob_0 - 1)

fidelity = st_post_processing(job_id='cqghp7g98a000084k70g')
print(np.sqrt(2*(1-fidelity)))