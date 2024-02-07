from AssetPricing import (GenerateEmpiricalProblems, 
                          Generate_D_Minus_E_problem, 
                          calculate_d_vector, 
                          StackEmpiricalProblems,
                          SolutionProjectionOpertator,
                          MultipleAbcissaObservable,
                          MultipleModelsObservable)
from enhanced_hybrid_hhl import HHL, QCL_QPE_IBM, HybridInversion, ideal_preprocessing, QuantumLinearSystemProblem, HHL_Result
from qiskit.quantum_info import Statevector

# from qiskit_ibm_provider import IBMProvider
from qiskit.providers.aer import AerSimulator

# provider = IBMProvider(instance='ibm-q-ncsu/nc-state/amplitude-estima')
# torino = provider.get_backend('ibm_torino')
simulator = AerSimulator()
# emulator = AerSimulator().from_backend(torino)

inversion_circuit = HybridInversion
clock = 3

task2_problem = GenerateEmpiricalProblems('IES', 2, 4)
d_vector = calculate_d_vector(4)
d_operator = Statevector(d_vector).to_operator()
maximum_eigenvalue = max(ideal_preprocessing(task2_problem)[0], key=abs)

eigenvalue_preprocessing = QCL_QPE_IBM(clock, simulator, max_eigenvalue=maximum_eigenvalue).estimate


task2_hhl = HHL(get_result_function='get_simulator_result',
                pre_processing=eigenvalue_preprocessing,
                eigenvalue_inversion=inversion_circuit,
                statevector = d_operator
                )

inner_product = task2_hhl.estimate(task2_problem,
                   3,
                   maximum_eigenvalue)

print(inner_product)