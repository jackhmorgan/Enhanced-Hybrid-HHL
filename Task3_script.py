import numpy as np
import os
from AssetPricing import (Generate_D_Minus_E_problem,
                          calculate_d_vector)

from qiskit.quantum_info import Statevector
from enhanced_hybrid_hhl import QuantumLinearSystemSolver
import matplotlib.pyplot as plt

file_name = 'benchmark_model_10_CRRA.txt'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)

benchmark_model = np.loadtxt(file_path).view(complex)

utility_function = 'IES'
gamma = 10
p_list = list(np.linspace(0,1,10))

d_vector = np.kron([[0],[1]], (calculate_d_vector(4)))
s1_vector = Statevector(d_vector)

s2_vector = Statevector(benchmark_model)

s3_problem = Generate_D_Minus_E_problem(utility_function=utility_function, gamma=gamma, size=4)
s3_vector = QuantumLinearSystemSolver(s3_problem).ideal_x_statevector

a3 = np.arccos(np.real(s3_vector.inner(s1_vector)))
ab = np.arccos(np.real(s2_vector.inner(s1_vector)))

s3 = 2 - (2*np.cos(a3))
sb = 2 - (2*np.cos(ab))

s1_observable = s1_vector.to_operator()
s2_observable = s2_vector.to_operator()

utilities = []
for p in p_list:
    observable = ((1-p)*s1_observable)+(p*s2_observable)
    utility = s3_vector.expectation_value(observable)
    utility *= s3/sb
    utilities.append(utility)

plt.plot(p_list,utilities)
plt.title('Classical Ambiguity Utility Function: '+utility_function+" Gamma: "+str(gamma))
plt.ylabel(r'$<d-\bar{v}(\theta))| (1-p)\text{P}_d + p \text{P}_B |d-\bar{v}(\theta))>$', fontsize=16)
plt.xlabel("p")
plt.show()
