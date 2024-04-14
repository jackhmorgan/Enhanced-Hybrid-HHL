import __future__

import sys
import os

# Assuming the parent directory of the tests folder is in your project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from .quantum_linear_system import QuantumLinearSystemProblem as QLSP
from qiskit.quantum_info import Statevector

def GenerateEmpiricalProblems(utility_function: str, gamma:int, size:int = 4):
    """
    The function `GenerateEmpiricalProblems` generates a quantum linear system problem for the standard
    Asset Pricing application with a given utility function and gamma value.
    
    :param utility_function: A string specifying either 'IES' or 'CRRA'.
    :param gamma: The model parameter gamma.
    :param size: The size parameter determines the number of regime states used in the simulation.
    :return: an instance of the Quatnum Linear System Problem class, which is initialized with the
    Hermitian matrix "hermitian" and the vector "b".
    """
    # Read classical simulation results.
    script_directory = os.path.dirname(os.path.realpath(__file__)) 
    subfolder_name = "C_matrix_"+str(size)
    file_name = utility_function+'_'+str(gamma)+'.xlsx'
    file_path = os.path.join(script_directory, subfolder_name, file_name)
    c_df = pd.read_excel(file_path, header=None)
    c_mat = np.asmatrix(c_df.to_numpy())

    size = c_mat.shape[0] # The size of the not hermitian matrix A

    # Create a diagonal block matrix with blocks c_mat and c_mat^dagger
    hermitian = np.zeros((2*size,2*size))

    for index, entry in np.ndenumerate(c_mat):
        hermitian[index[0],size+index[1]] = entry 

    for index, entry in np.ndenumerate(c_mat.H):
        hermitian[size+index[0],index[1]] = entry

    # define |b> unit vector
    unit = np.ones((size,1))
    unit /= np.linalg.norm(unit)
    b = np.kron([[1],[0]], unit)

    # Create Quantum Linear System Problem
    problem_c = QLSP(A_matrix = hermitian,b_vector = b)
    return problem_c

def StackEmpiricalProblems(problem_list: list[QLSP]):
    """
    The function takes a list of QLSP problems and stacks their A_matrices and b_vectors into a single
    QLSP problem.
    
    :param problem_list: The `problem_list` parameter is a list of QLSP (Quantum Linear System Problem)
    objects. Each object represents a specific problem with its own A_matrix and b_vector
    :type problem_list: list[QLSP]
    :return: an instance of the QLSP class with the stacked A_matrix and b_vector.
    """

    n_problems = len(problem_list)
    c_rows = []
    b_stacked = None
    shape = problem_list[0].A_matrix.shape
    blank_row = [np.zeros(shape) for _ in range(n_problems)]

    for i, problem in enumerate(problem_list):
        row = blank_row.copy()
        matrix = problem.A_matrix
        row[i] = matrix
        c_rows.append(row)
        if i==0:
            b_stacked = problem.b_vector
        else:    
            b_stacked = np.vstack((b_stacked, problem.b_vector))
    c_stacked = np.block(c_rows)
    
    
    return QLSP(A_matrix=c_stacked, b_vector=b_stacked) 

def calculate_d_vector(size: int):
    """
    The function `calculate_d_vector` reads the mean divideng growth from simulation results.
    
    :param size: The parameter "size" represents the number of states used for the simulation
    :type size: int
    :return: The function `calculate_d_vector` returns a numpy array `d_full` which contains the
    calculated values based on the given inputs.
    """

    script_directory = os.path.dirname(os.path.realpath(__file__)) 
    subfolder_name = "All_spreadsheet_"+str(size)
    file_name = '1.xlsx'
    file_path = os.path.join(script_directory, subfolder_name, file_name)

    ### read evenly-spaced discretiztion of empirical de-meaned CDF of div growth
    input_1 = pd.read_excel(file_path,header=None)
    states_empirical_CDF = input_1.iloc[:-1,0].values

    ### read mean for div growth
    div_growth_mean = input_1.iloc[-1,0]

    d_half = states_empirical_CDF + div_growth_mean

    size = d_half.shape[0]
    d_full = np.zeros((2*size,1))

    ### average growth over half time steps
    for i, observation in enumerate(d_half):
        index = 2*i
        d_full[index] = observation
        if i == size-1:
            d_full[index+1] = observation
        else:
            ave = observation + d_half[i+1]
            ave /= 2
            d_full[index+1] = ave
            
    return d_full

def stack_vector(n_models: int, d_vector: ArrayLike):
    """
    The function `stack_vector` takes in the number of models `n_models` and a vector `d_vector`, and
    returns a normalized version of d_vector stacked n_models number of times.
    
    :param n_models: The number of models you want to stack the vector for
    :param d_vector: The `d_vector` parameter is a 1-dimensional numpy array representing a vector
    :return: a list of normalized vectors.
    """

    d = np.kron([[0],[1]], d_vector)
    
    # Use numpy.tile to replicate the vector n times
    stacked_vector = np.tile(d, (n_models, 1))

    # Use numpy.vstack to stack the replicated vectors vertically
    result_matrix = np.vstack(stacked_vector)
    norm = np.linalg.norm(result_matrix)
    result_normed = result_matrix/norm
    result_list = list(result_normed.T[0])
    return result_list

def Generate_D_Minus_E_problem(utility_function: str, gamma: int, size: int):
    """
    The function `Generate_D_Minus_E_problem` generates a Quantum Linear System Problem to solve for the
    vector |d-e>.
    
    :param utility_function: A string specifying either 'IES' or 'CRRA'.
    :param gamma: The model parameter gamma.
    :param size: The size parameter determines the number of regime states used in the simulation.
    :return: an instance of the QLSP (Quadratic Linear Sum Problem) class.
    """
    d_vector = calculate_d_vector(size)

    ep = GenerateEmpiricalProblems(utility_function, gamma, size)
    d_norm = d_vector/np.linalg.norm(d_vector)

    b_unit = ep.b_vector[:int(4*size/2)]
    c_mat = ep.A_matrix[int(4*size/2):,:int(4*size/2)]

    b = np.dot(c_mat,d_norm) - b_unit
    b_stacked = np.vstack((b, np.zeros((len(b),1))))
    return QLSP(A_matrix=ep.A_matrix, b_vector=b_stacked)


