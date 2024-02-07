from enhanced_hybrid_hhl import RandomQLSP, QCL_QPE_IBM, HybridInversion, HHL, ideal_preprocessing

test = RandomQLSP(2,5)

from qiskit_ibm_provider import IBMProvider
from qiskit.providers.aer import AerSimulator

provider = IBMProvider(instance='ibm-q-ncsu/nc-state/amplitude-estima')
torino = provider.get_backend('ibm_torino')
simulator = AerSimulator()
emulator = AerSimulator().from_backend(torino)

ideal_eigenvalues, ideal_projections = ideal_preprocessing(problem=test)
qpe = QCL_QPE_IBM(3, simulator, max_eigenvalue=max(ideal_eigenvalues, key=abs))

pre_processing = qpe.estimate(problem=test)
print(pre_processing)
#e_h_hhl = HHL(get_result='simulator', pre_processing=pre_processing, eigenvalue_inversion=HybridInversion)
