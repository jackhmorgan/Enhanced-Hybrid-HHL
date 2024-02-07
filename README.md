# Enhanced-Hybrid-HHL
This project contains the framework to run the HHL, Hybrid HHL, and enhanced Hybrid HHL algorithms.
The variant of the algorithm is determined by the choice of inversion circuit and 
eigenvalue preprocessing parameter.

## Example Implementation 
### Step one: Define a Quantum Linear System Problem
In this example we estimate $\ket{x}$ in the equation $\mathcal{A} \ket{x} = \ket{b}$
where $\mathcal{A} = $ A_matrix and $\ket{b} =$ b_vector defined below. We test the accuracy
of the algorithm by observing the estimated state $\ket{x}$ with the projection operator 
onto the ideal solution.

```python
from enhanced_hybrid_hhl import QuantumLinearSystemProbelm
import numpy as np

# Define linear system
A_matrix = np.asmatrix([[0.5, 0.33],[0.33, 0.5]])
b_vector = [1,0]

problem = QuantumLinearSystemProblem(A_matrix, b_vector)
```
### Step two: Choose the algorithm parameters
```python
from enhanced_hybrid_hhl import QCL_QPE_IBM, HybridInversion, QuantumLinearSystemSolver

from qiskit_aer import AerSimulator

eigenvalue_precision = 6 # number of bits of the estimates
simulator = AerSimulator() # backend to run the circuit

# create instance of QCL_QPE class
preprocessing_algorithm = QCL_QPE_IBM(eigenvalue_precision,
 simulator)

# define the function that will be used by the HHL class
preprocessing_function = preprocessing_algorithm.estimate
inversion = HybridInversion

# define operator to observe estimated solution.
ideal_x_observable = QuantumLinearSystemSolver(example_problem).ideal_x_statevector.to_operator()
```

### Step three: Create HHL class
```python
from enhanced_hybrid_hhl import HHL

enhanced_hybrid_hhl  = HHL('get_simulator_result',
          preprocessing_function,
          inversion,
          backend=simulator,
          statevector=ideal_x_observable)
```
### Step four: run the algorithm
```python
hhl_result = hhl.estimate(problem)

print(hhl_result)
```
## License

This project uses [Apache 2.0 License]([url](https://github.com/jackhmorgan/Enhanced-Hybrid-HHL/blob/main/LICENSE)https://github.com/jackhmorgan/Enhanced-Hybrid-HHL/blob/main/LICENSE)









