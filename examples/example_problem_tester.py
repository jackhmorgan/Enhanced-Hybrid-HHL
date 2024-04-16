# This Python script is performing the following tasks:
#   - Import a json problem from the specified file 
import json
import numpy as np

filename = 'examples/example_problems_N4.json'

from enhanced_hybrid_hhl import (ExampleQLSP,
                                 ideal_preprocessing,
                                 QuantumLinearSystemProblem)
with open(filename, 'r') as f:
    problem_list = json.load(f)

for json_problem in problem_list:
    A_matrix = json_problem['A_matrix']
    b_vector = json_problem['b_vector']

    problem = QuantumLinearSystemProblem(A_matrix = A_matrix, b_vector=b_vector)


    eigenvalues, projections = ideal_preprocessing(problem)

    print('eigenvalues : ',eigenvalues)
    print('projections : ',projections)