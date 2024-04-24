from AssetPricing import (GenerateEmpiricalProblems, 
                          Generate_D_Minus_E_problem, 
                          calculate_d_vector, 
                          StackEmpiricalProblems,
                          SolutionProjectionOpertator,
                          MultipleAbcissaObservable,
                          MultipleModelsObservable)
from enhanced_hybrid_hhl import (HHL, 
                                 GrayCodeInversion, 
                                 QuantumLinearSystemProblem, 
                                 HHL_Result,
                                 QuantumLinearSystemSolver,
                                )
from qiskit.quantum_info import Statevector

utility_function_list = ['IES', 'CRRA']
gamma_list = [2,5,10]

for uf in utility_function_list:
    for g in gamma_list:
        utility_function = uf
        gamma = g
        v_problem = GenerateEmpiricalProblems(utility_function=utility_function,
                                            gamma=gamma,
                                            size=4)

        dv_problem = Generate_D_Minus_E_problem(utility_function=utility_function,
                                                gamma=gamma,
                                                size=4)

        v_vector = QuantumLinearSystemSolver(v_problem).ideal_x_statevector
        dv_vector = QuantumLinearSystemSolver(dv_problem).ideal_x_statevector

        print(v_vector.inner(dv_vector))

        d_vector = calculate_d_vector(size=4)
        d_vector