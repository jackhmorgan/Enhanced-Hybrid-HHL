import pandas as pd
import numpy as np

from QuantumLinearSystemsProblem import QuantumLinearSystemsProblem as QLSP

def GenerateEmpiricalProblems(utility_function, gamma, size=4):
    path=''
    folder = path+'./C_matrix_'+str(size)+'/'
    matrix_name = utility_function+'_'+str(gamma)
    c_df = pd.read_excel(folder+matrix_name+'.xlsx', header=None)
    c_mat = np.asmatrix(c_df.to_numpy())
    #c_mat_norm = np.linalg.norm(c_mat)
    #c_mat /= c_mat_norm 

    size = c_mat.shape[0]

    hermitian = np.zeros((2*size,2*size))

    for index, entry in np.ndenumerate(c_mat):
        hermitian[index[0],size+index[1]] = entry
        #print(hermitian[index[0],size+index[1]])

    for index, entry in np.ndenumerate(c_mat.H):
        hermitian[size+index[0],index[1]] = entry


    unit = np.ones((size,1))
    unit /= np.linalg.norm(unit)
    b = np.kron([[1],[0]], unit)
    problem_c = QLSP(A = hermitian,b = b)
    return problem_c