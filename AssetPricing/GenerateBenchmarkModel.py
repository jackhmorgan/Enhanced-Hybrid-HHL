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
from .quantum_linear_system import QuantumLinearSystemSolver
from qiskit.quantum_info import Statevector

def C_Mat_abscissa(i, utility_function, gamma, size):
    ### read evenly-spaced discretiztion of empirical de-meaned CDF of div growth
    script_directory = os.path.dirname(os.path.realpath(__file__)) 
    subfolder_name = "All_spreadsheet_"+str(size)
    file_name = '1.xlsx'
    file_path = os.path.join(script_directory, subfolder_name, file_name)
    path = os.path.join(script_directory, subfolder_name)

    input_1 = pd.read_excel(file_path,header=None)
    def states_from_simulations(i):
        # set i from 1...1000
        return input_1.iloc[:-1,i+1].values
    def states_to_dividend_growth(states : np.array, a = 0.01037):
        dividend_growth = np.zeros((len(states),1))
        for indx, element in np.ndenumerate(states):
            dividend_growth[indx] += np.exp(a+element)
        return dividend_growth
    def growth_to_H(dividend_growth, b=0.03630):
        b_matrix = np.array([[np.exp(b), np.exp(-b)],
                            [np.exp(b), np.exp(-b)]])
        one = np.ones((1,len(dividend_growth)))
        H = np.kron(np.matmul(dividend_growth,one),
                        b_matrix)
        return H
    def sdf_equation(state):
        return np.exp(-0.8970 + (1.2038*state))

    def states_to_sdf(states : np.array, sdf_equation : callable):
        sdf = np.zeros((len(states),1))
        for indx, element in np.ndenumerate(states):
            sdf[indx] += sdf_equation(element)
        return sdf
    
    def parameters_to_xi(gamma, function, b=0.03630):

        if function == 'CRRA':
            return -b*gamma
        if function == 'IES':
            return  0.0412 - (0.0775*gamma)
        else:
            raise Exception("Function must be either 'CRRA' or 'IES'")
    
    def sdf_to_M(sdf: np.array, xi):
        xi_matrix = np.array([[np.exp(xi), np.exp(-xi)],
                            [np.exp(xi), np.exp(-xi)]])
        one = np.ones((1,size))
        M = np.kron(np.matmul(sdf,one),
                        xi_matrix)
        
        return M
    
    def Psi_and_Pi_to_B(Psi, Pi):
        size = Psi.shape[0]
        B = np.zeros((size,size))
        for row in range(size):
            for pi, psi in zip(Pi[row], Psi[row]):
                B[row][row] += pi*psi
                
            B[row][row] /= size
        return B
    
    def H_M_and_Pi_to_A(H, M, Pi):

        I = np.identity(H.shape[0])
        A = np.multiply(H, M)
        A = np.multiply(A, Pi)
        A = I - A
        return A
    def trans_matrix_from_simulations(i):
        # set i from 1...1000
        return pd.read_excel(path+'/'+str(i+3)+'.xlsx',header=None).values

    states_i = states_from_simulations(i) 
    dividend_growth = states_to_dividend_growth(states_i)
    H = growth_to_H(dividend_growth)
    sdf = states_to_sdf(states_i, sdf_equation)
    xi = parameters_to_xi(gamma, utility_function, b=0.03630)
    M = sdf_to_M(sdf, xi)
    
    Pi = np.kron(trans_matrix_from_simulations(i), 
            np.array([[1/2, 1/2],[1/2, 1/2]]))
    Psi = np.multiply(H, M)
    B = Psi_and_Pi_to_B(Psi, Pi)
    
    A = H_M_and_Pi_to_A(H, M, Pi)
    
    C = np.matmul(np.linalg.inv(B),A)
    return C

def GenerateAbscissaProblems(i, utility_function: str, gamma:int, size:int = 4):
    """
    The function `GenerateEmpiricalProblems` generates a quantum linear system problem for the standard
    Asset Pricing application with a given utility function and gamma value.
    
    :param utility_function: A string specifying either 'IES' or 'CRRA'.
    :param gamma: The model parameter gamma.
    :param size: The size parameter determines the number of regime states used in the simulation.
    :return: an instance of the Quatnum Linear System Problem class, which is initialized with the
    Hermitian matrix "hermitian" and the vector "b".
    """
    c_mat = np.asmatrix(C_Mat_abscissa(i, utility_function, gamma, size))

    # Create a diagonal block matrix with blocks c_mat and c_mat^dagger
    hermitian = np.zeros((4*size,4*size))

    for index, entry in np.ndenumerate(c_mat):
        hermitian[index[0],2*size+index[1]] = entry 

    for index, entry in np.ndenumerate(c_mat.H):
        hermitian[2*size+index[0],index[1]] = entry

    # define |b> unit vector
    unit = np.ones((2*size,1))
    unit /= np.linalg.norm(unit)
    b = np.kron([[1],[0]], unit)

    # Create Quantum Linear System Problem
    problem_c = QLSP(A_matrix = hermitian,b_vector = b)
    return problem_c

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

def Generate_D_Minus_E_Abscissa_problem(i, utility_function: str, gamma: int, size: int):
    """
    The function `Generate_D_Minus_E_problem` generates a Quantum Linear System Problem to solve for the
    vector |d-e>.
    
    :param utility_function: A string specifying either 'IES' or 'CRRA'.
    :param gamma: The model parameter gamma.
    :param size: The size parameter determines the number of regime states used in the simulation.
    :return: an instance of the QLSP (Quadratic Linear Sum Problem) class.
    """
    d_vector = calculate_d_vector(size)

    ep = GenerateAbscissaProblems(i,utility_function, gamma, size)
    d_norm = d_vector/np.linalg.norm(d_vector)

    b_unit = ep.b_vector[:int(4*size/2)]
    c_mat = ep.A_matrix[int(4*size/2):,:int(4*size/2)]

    b = np.dot(c_mat,d_norm) - b_unit
    b_stacked = np.vstack((b, np.zeros((len(b),1))))
    return QLSP(A_matrix=ep.A_matrix, b_vector=b_stacked)

def GenerateBenchmarkModel(utility_function, gamma, size):
    script_directory = os.path.dirname(os.path.realpath(__file__)) 
    subfolder_name = 'All_spreadsheet_'+str(size)
    file_name = '1.xlsx'
    file_path = os.path.join(script_directory, subfolder_name, file_name)
    input_1 = pd.read_excel(file_path,header=None)

    ### read KL_i,i=1...1000
    KL = input_1.iloc[-1,2:].values

    total_weight = np.sum(KL)
    state_list = []

    for i in range(1,1000):
        problem = Generate_D_Minus_E_Abscissa_problem(i, utility_function, gamma, size)
        state = QuantumLinearSystemSolver(problem).ideal_x_statevector
        state_list.append(state)
        
        
    weight_list = [weight/total_weight for weight in KL]
    final_state = None
    for weight, statevector in zip(weight_list, state_list):
        if final_state == None:
            final_state = weight*statevector
        else:
            final_state += weight*statevector

    return final_state

