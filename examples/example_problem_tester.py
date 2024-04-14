import json
import numpy as np

filename = 'examples/example_problem_16_2.json'

import sys
sys.path.append('C:/Users/19899/Documents/HHL/HybridInversion/Enhanced-Hybrid-HHL')

from enhanced_hybrid_hhl import (ExampleQLSP,
                                 Iterative_QPE_Preprocessing,
                                 QuantumLinearSystemProblem)
with open(filename, 'r') as f:
    data = json.load(f)

A_matrix = data['A_matrix']
b_vector = data['b_vector']

#print(np.linalg.eigh(A_matrix))

problem = QuantumLinearSystemProblem(A_matrix = A_matrix, b_vector=b_vector)
#problem = ExampleQLSP(0.33)

from qiskit_ibm_runtime import Session, QiskitRuntimeService

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-ncsu/nc-state/amplitude-estima',
    token='35e7316e04600dcc5e1b36a25b8eb9c61601d1456fe4d0ddb1604a8ef1847f0614516de700c3f5d9bac07141ade760115a07f1a706307d04fd16066d1a0dd109'
)

backend = service.get_backend("ibmq_qasm_simulator")

with Session(service=service, backend=backend) as session:

    preprocessing = Iterative_QPE_Preprocessing(clock=3,
                                                session=session)

    eigenvalues, projections = preprocessing.estimate(problem)

print(eigenvalues)
print(projections)