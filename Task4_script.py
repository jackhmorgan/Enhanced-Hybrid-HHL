import numpy as np
import os
import json
from AssetPricing import (Generate_D_Minus_E_problem,
                          GenerateEmpiricalProblems,
                          GenerateBenchmarkModel,
                          calculate_d_vector)
from enhanced_hybrid_hhl import QuantumLinearSystemSolver
from qiskit.quantum_info import Statevector

file_name = 'benchmark_model_10_CRRA.txt'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)


#np.savetxt(file_path, data)

loaded_data = np.loadtxt(file_path).view(complex)
statevector = GenerateBenchmarkModel(utility_function='CRRA',
                                     gamma=10,
                                     size = 4)
data = statevector.data
np.savetxt(file_path, data.view(float))

same = np.isclose(data, loaded_data)
print(same)
#d_vector = np.kron([[0],[1]], (calculate_d_vector(4)))
#s1_vector = Statevector(d_vector)

#s2_problem = Generate_D_Minus_E_problem(utility_function='CRRA', gamma=10, size=4)
#s2_vector = QuantumLinearSystemSolver(s2_problem).ideal_x_statevector

#s3_problem = Generate_D_Minus_E_problem(utility_function='IES', gamma=2, size=4)
#s3_vector = QuantumLinearSystemSolver(s3_problem).ideal_x_statevector

#def QuantumAmbiguity(s1, s2, alpha, delta):
#    return (alpha*s1) + np.exp(delta*1j)*np.sqrt(1-(alpha**2))*s2

#a= 0.9
#for d in np.linspace(0,np.pi,5):
#    s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=d)
#    print('delta = ',d,' Utility = ',s12_vector.inner(s3_vector))