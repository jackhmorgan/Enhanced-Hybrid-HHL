# Enhanced-Hybrid-HHL
This project contains code and example problems used to the Enhanced Hybrid HHL Algorithm [1]. For benchmarking purposes, the project can implement the standard HHL algorithm [2], the Hybrid HHL algorithm [3], as well as the variation of the Hybrid HHL algorithm proposed by [4]. The variant of the HHL class is determined by the choice of inversion circuit and 
eigenvalue preprocessing parameter. 

## Example Implementation 
### Step one: Define a Quantum Linear System Problem
In this example we estimate $\ket{x}$ in the equation $\mathcal{A} \ket{x} = \ket{b}$
where $\mathcal{A} = $ A_matrix and $\ket{b} =$ b_vector defined below. We test the accuracy
of the algorithm by observing the estimated state $\ket{x}$ with the projection operator 
onto the ideal solution.

```python
from enhanced_hybrid_hhl import (HHL, 
                                 Lee_preprocessing,  
                                 HybridInversion, 
                                 QuantumLinearSystemProblem, 
                                 QuantumLinearSystemSolver,
                                 EnhancedHybridInversion)
import numpy as np
from qiskit_aer import AerSimulator

# define the backend to run the circuits on
simulator = AerSimulator()

# Define quantum linear system problem to be solved with HHL
a_matrix = np.array([[ 0.5 , -0.25],
        [-0.25,  0.5 ]])
b_vector = np.array([[1.], [0.]])
problem = QuantumLinearSystemProblem(A_matrix=a_matrix,
                                     b_vector=b_vector)
```
### Step two: Choose the algorithm parameters
```python
k = 3 # clock qubits for hhl.
l = k+2 # clock qubits for enhanced preprocessing.
min_prob = 2**-k # hybrid preprocessing relevance threshold.
relevance_threshold = 2**-l # enhanced hybrid preprocessing relevance threshold.
maximum_eigenvalue = 1 # Over estimate of largest eigenvalue in the system.

get_result_type = 'get_swap_test_result'
ideal_x_statevector = QuantumLinearSystemSolver(problem=problem).ideal_x_statevector
```

### Step three: Define Preprocessing and Inversion circuit classes
```python
# In this example, we use the standard QPEA used by Lee et al.
enhanced_preprocessing = Lee_preprocessing(num_eval_qubits=l,
                                  max_eigenvalue= maximum_eigenvalue, 
                                  backend=simulator).estimate

enhanced_eigenvalue_inversion = EnhancedHybridInversion
```
### Step four: Create the HHL Class
```python
enhanced_hybrid_hhl = HHL(get_result_function= get_result_type,
          preprocessing= enhanced_preprocessing,
          eigenvalue_inversion= enhanced_eigenvalue_inversion,
          backend=simulator,
          statevector=ideal_x_statevector)
```
### Step five: Run the algorithm
```python
enhanced_hybrid_hhl_result = enhanced_hybrid_hhl.estimate(problem=problem,
                                                          num_clock_qubits=k,
                                                          max_eigenvalue=1)

print(enhanced_hybrid_hhl_result)
```
### References
<a id="1">[1]</a> 
Morgan, J., Ghysels, E., & Mohammadbagherpoor, H. (2024). An Enhanced Hybrid HHL Algorithm. arXiv preprint arXiv:2404.10103.

<a id="2">[2]</a> 
Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for linear systems of equations. Physical review letters, 103(15), 150502.

<a id="3">[3]</a> 
Lee, Y., Joo, J., & Lee, S. (2019). Hybrid quantum linear equation algorithm and its experimental test on IBM Quantum Experience. Scientific reports, 9(1), 4778.

<a id="4">[4]</a> 
Yalovetzky, R., Minssen, P., Herman, D., & Pistoia, M. (2021). Hybrid HHL with Dynamic Quantum Circuits on Real Hardware. arXiv preprint arXiv:2110.15958.

## License

This project uses [Apache 2.0 License]([url](https://github.com/jackhmorgan/Enhanced-Hybrid-HHL/blob/main/LICENSE)https://github.com/jackhmorgan/Enhanced-Hybrid-HHL/blob/main/LICENSE)









