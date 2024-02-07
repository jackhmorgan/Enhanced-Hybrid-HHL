import numpy as np
from qiskit.quantum_info.operators import Operator

import numpy as np
import pandas as pd
import os

from .GenerateEmpiricalProblems import GenerateEmpiricalProblems
from .quantum_linear_system import QuantumLinearSystemSolver

def MultipleAbcissaObservable(size, probability):
    script_directory = os.path.dirname(os.path.realpath(__file__)) 
    subfolder_name = 'All_spreadsheet_'+str(size)
    file_name = '1.xlsx'
    file_path = os.path.join(script_directory, subfolder_name, file_name)
    input_1 = pd.read_excel(file_path,header=None)

    states_empirical_CDF = input_1.iloc[:-1,0].values

    ### read quadrature abscissa for theta_hat
    states_theta_hat = input_1.iloc[:-1,1].values

    ### read quadrature abscissa for theta_i,i=1...1000
    def states_from_simulations(i):
        # set i from 1...1000
        return input_1.iloc[:-1,i+1].values

    ### read mean for div growth
    div_growth_mean = input_1.iloc[-1,0]

    ### read KL_i,i=1...1000
    KL = input_1.iloc[-1,2:].values

    summation = 0
    for i in range(1,1000):
        summation += (1+KL[i])**-1
    
    for i in range(1,1000):
        state = states_from_simulations(i)
        state /= np.linalg.norm(state)
        projection = np.matmul(np.matrix(state).T, np.matrix(state))
        weight = 1/((1+KL[i])*summation)

        if i==1:
            observable_matrix_sum = weight*projection
        else: 
            observable_matrix_sum += weight*projection

    state = states_from_simulations(0)
    projection_ve = np.matmul(np.matrix(state).T, np.matrix(state))

    observable_matrix = (1-probability)*projection_ve + probability*observable_matrix_sum

    return Operator(observable_matrix)

def MultipleModelsObservable(size, cdf_list, gamma_list, probability_list):

    observable_matrix = np.zeros((4*size,4*size))
    
    for cdf, gamma, probability in zip(cdf_list, gamma_list, probability_list):
        problem = GenerateEmpiricalProblems(utility_function=cdf, gamma=gamma, size=size)
        v_e = np.linalg.inv(problem.A_matrix).dot(problem.b_vector)
        projection = np.matmul(np.matrix(v_e).T, np.matrix(v_e))
        observable_matrix += probability*projection
    return Operator(observable_matrix)

def SolutionProjectionOpertator(problem):
    solution = QuantumLinearSystemSolver(problem)
    solution_vector = solution.ideal_x_statevector
    return solution_vector.to_operator()