from qiskit_ionq import IonQProvider
from qiskit import QuantumCircuit

ionq_provider = IonQProvider(token='eqMVRwVhZkVIX9lEBlIMYhxFtneQnCD5')

backend = ionq_provider.get_backend('ionq_simulator')

test_circuit = QuantumCircuit(3,0)

test_circuit.x(0)
test_circuit.measure_all()

backend.run(test_circuit, noise_model = "aria-1")