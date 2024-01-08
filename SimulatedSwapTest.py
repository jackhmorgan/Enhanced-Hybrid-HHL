from qiskit.quantum_info import Statevector as sv
import numpy as np
def SimulatedSwapTest(problem, results):
    
    x_sv = sv(list(problem.ideal_x()[0]))
    
    passed_results = {key[:-1] : value for key, value in results.items() if key[-1]=='1'}
    
    res_state = []
    total = sum(passed_results.values())
    results_dict = {int(key,2) : (counts/total) for (key, counts) in passed_results.items()}
    for i in range(len(x_sv.data)):
        if i in results_dict.keys():
            res_state.append(np.sqrt(results_dict[i]))
        else:
            res_state.append(0)

    res_sv = sv(res_state)
    
    return x_sv.inner(res_sv)