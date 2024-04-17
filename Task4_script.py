import numpy as np
import os
import json
from AssetPricing import (Generate_D_Minus_E_problem,
                          GenerateEmpiricalProblems,
                          GenerateBenchmarkModel,
                          calculate_d_vector)
from enhanced_hybrid_hhl import QuantumLinearSystemSolver
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

file_name = 'benchmark_model_10_CRRA.txt'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)


#np.savetxt(file_path, data)

benchmark_model = np.loadtxt(file_path).view(complex)
#statevector = GenerateBenchmarkModel(utility_function='CRRA',
#                                     gamma=10,
#                                     size = 4)
#data = statevector.data
#np.savetxt(file_path, data.view(float))

utility_function = 'IES'
gamma = 2

d_vector = np.kron([[0],[1]], (calculate_d_vector(4)))
s1_vector = Statevector(d_vector)

s2_vector = Statevector(benchmark_model)

s3_problem = Generate_D_Minus_E_problem(utility_function=utility_function, gamma=gamma, size=4)
s3_vector = QuantumLinearSystemSolver(s3_problem).ideal_x_statevector

a3 = np.arccos(np.real(s3_vector.inner(s1_vector)))
ab = np.arccos(np.real(s2_vector.inner(s1_vector)))

s3 = 2 - (2*np.cos(a3))
sb = 2 - (2*np.cos(ab))

print(s3/sb)

def QuantumAmbiguity(s1, s2, alpha, delta):
    return (alpha*s1) + np.exp(delta*1j)*np.sqrt(1-(alpha**2))*s2

alphas = list(np.linspace(0,1,5))
deltas = list(np.linspace(0,np.pi,10))
for a in alphas:
    utilities = []
    for d in deltas:
        s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=d)
        s12_operator = s12_vector.to_operator()
        utility = s3_vector.expectation_value(s12_operator)
        utility *= s3/sb
        utilities.append(abs(utility))
        #print('delta = ',d,' Utility = ',s3_vector.expectation_value(s12_operator))
    plt.plot(deltas, utilities, label='alpha = '+str(a))
plt.title('Utility Function: '+utility_function+" Gamma: "+str(gamma))
plt.xlabel('delta')
plt.ylabel(r'$\langle \mathrm{tr}(P|S1,2(\alpha,\delta)\rangle P3)$', fontsize=16)
plt.legend()
plt.show()