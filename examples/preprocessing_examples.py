import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')

from enhanced_hybrid_hhl import (HHL, 
                                 EnhancedHybridInversion,
                                 HybridInversion,
                                 CannonicalInversion,
                                 Lee_preprocessing,
                                 Yalovetsky_preprocessing,
                                 ideal_preprocessing,
                                 QuantumLinearSystemProblem,
                                 ExampleQLSP,
                                 QuantumLinearSystemSolver)
from qiskit_ibm_runtime import QiskitRuntimeService, Session

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-ncsu/nc-state/amplitude-estima',
    token='35e7316e04600dcc5e1b36a25b8eb9c61601d1456fe4d0ddb1604a8ef1847f0614516de700c3f5d9bac07141ade760115a07f1a706307d04fd16066d1a0dd109'
)

backend = service.get_backend('ibm_torino')


import json

k = 3
l = k+2
file_path = 'examples/example_problem.json'
with open(file_path, "r") as json_file:
    data = json.load(json_file)

problem = QuantumLinearSystemProblem(data['A_matrix'], data['b_vector'])

with Session(service=service, backend=backend) as session:
    preprocessing=Lee_preprocessing(num_eval_qubits=k,  
                                        max_eigenvalue=1,
                                        session=session,
                                        wait_for_result=False
                                        )
    preprocessing.estimate(problem)
    e_preprocessing = Lee_preprocessing(num_eval_qubits=l,  
                                        max_eigenvalue=1,
                                        session=session,
                                        wait_for_result=False
                                        )
    e_preprocessing.estimate(problem)