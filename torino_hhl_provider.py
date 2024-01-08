from qiskit_ibm_provider import IBMProvider
from qiskit import transpile

# Save your credentials on disk.
# IBMProvider.save_account(token='<IBM Quantum API key>')

provider = IBMProvider(instance='ibm-q-ncsu/nc-state/amplitude-estima')
# Or save your credentials on disk.
# QiskitRuntimeService.save_account(channel='ibm_quantum', instance='ibm-q-ncsu/nc-state/amplitude-estima', token='<IBM Quantum API key>')

backend = provider.get_backend('ibm_torino')

from GenerateEmpiricalProblems import GenerateEmpiricalProblems as EmpiricalProblem
from E_Hybrid_HHL_torino import E_Hybrid_HHL

circuits = []
for gamma in [2,5,10]:
    
    for sdf in ['IES', 'CRRA']:
        clock = 3
        problem = EmpiricalProblem(sdf,gamma,8)
        max_eigenvalue = max(problem.ideal_eigen().keys(), key = abs) #maximum eigenvalue, as determined by the long run risk
        scale = abs((0.5-2**-clock)/max_eigenvalue)
        test_HHL = E_Hybrid_HHL(problem=problem)
        hhl_circuit = test_HHL.construct_circuit(egn=problem.ideal_eigen(), 
                                         clock=clock,
                                         max_eigen = max_eigenvalue)
        hhl_circuit.name = sdf+', '+str(gamma)
        transpiled_circuit = transpile(hhl_circuit, backend)
        print(transpiled_circuit.depth())
        result = backend.run(transpiled_circuit).result()
        print(result.get_distribution)
        
