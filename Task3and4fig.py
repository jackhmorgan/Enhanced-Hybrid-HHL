import numpy as np
import os
import json
from AssetPricing import (Generate_D_Minus_E_problem,
                          GenerateEmpiricalProblems,
                          GenerateBenchmarkModel,
                          calculate_d_vector)
from enhanced_hybrid_hhl import (QuantumLinearSystemSolver,
                                 QuantumLinearSystemProblem)
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

file_name = 'benchmark_model_10_CRRA.txt'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)

benchmark_model = np.loadtxt(file_path).view(complex)

utility_function = 'IES'
gamma = 2

# Classical Ambiguity
d_vector = np.kron([[0],[1]], (calculate_d_vector(4)))
#d_vector /= np.linalg.norm(d_vector)
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

#Quantum Ambiguity
a3 = np.arccos(np.real(s3_vector.inner(s1_vector)))
ab = np.arccos(np.real(s2_vector.inner(s1_vector)))

benchmark_dv = s2_vector - s1_vector


benchmark_utility = (s2_vector).inner(benchmark_dv)

s3 = 2 - (2*np.cos(a3))
sb = 2 - (2*np.cos(ab))

def QuantumAmbiguity(s1, s2, alpha, delta):
    return (np.sqrt(1-(alpha**2))*s1) + np.exp(delta*1j)*alpha*s2

probs = list(np.linspace(0,1,10)[0:-1])

classical_utilities = []
upper_bounds = []
lower_bounds = []

classical_operators = []
quantum_upper_bound_operators = []
quantum_lower_bound_operators = []

deltas = list(np.linspace(0,np.pi,10))
quantum_utilities = {key : [] for key in deltas}
for p in probs:
    # Classical
    a = np.sqrt(p)
    observable = ((1-p)*s1_observable)+(p*s2_observable)
    classical_operators.append(observable)
    utility = s3_vector.expectation_value(observable.data)
    utility *= s3/sb
    classical_utilities.append(utility)

    # Upper bound
    s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=0)
    s12_operator = s12_vector.to_operator()
    quantum_upper_bound_operators.append(s12_operator.data)
    utility = s3_vector.expectation_value(s12_operator)
    utility *= s3/sb
    upper_bounds.append(abs(utility))

    # Lower bound
    s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=np.pi)
    s12_operator = s12_vector.to_operator()
    quantum_lower_bound_operators.append(s12_operator.data)
    utility = s3_vector.expectation_value(s12_operator)
    utility *= s3/sb
    lower_bounds.append(abs(utility))

    for d in deltas:
        s12_vector = QuantumAmbiguity(s1=s1_vector, s2=s2_vector, alpha=a, delta=d)
        s12_operator = s12_vector.to_operator()
        utility = s3_vector.expectation_value(s12_operator)
        utility *= s3/sb
        quantum_utilities[d].append(abs(utility))

# Save the lists to files
#np.save('classical_operators.npy', classical_operators)
#np.save('quantum_upper_bound_operators.npy', quantum_upper_bound_operators)
#np.save('quantum_lower_bound_operators.npy', quantum_lower_bound_operators)
np.savez(utility_function+'_'+str(gamma)+'tasks_3_4.npz', 
         classical_operators=classical_operators, 
         quantum_upper_bound_operators=quantum_upper_bound_operators, 
         quantum_lower_bound_operators=quantum_lower_bound_operators,
         benchmark_model=benchmark_model.data)

for delta, list in quantum_utilities.items():
    plt.plot(probs, list,'--', label = delta)
plt.plot(probs, classical_utilities, label='classical')
plt.plot(probs, upper_bounds, label = 'upper')
plt.plot(probs, lower_bounds, label = 'lower')
plt.title('Utility Function: '+utility_function+" Gamma: "+str(gamma))
plt.hlines(benchmark_utility, 0, 1, label='benchmark?')
plt.xlabel('probability')
plt.ylabel(r'$\langle \mathrm{tr}(P|S1,2(\alpha,\delta)\rangle P3)$', fontsize=16)
plt.legend()
plt.show()