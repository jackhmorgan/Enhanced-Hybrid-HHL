from qiskit import QuantumCircuit

class SwapTest(QuantumCircuit):
    def __init__(self,
                 num_state_qubits):
        num_qubits = 2*num_state_qubits+1
        super().__init__(num_qubits)
        self.h(-1)
        for i in range(num_state_qubits):
            self.cswap(-1,i,num_state_qubits+i)
        self.h(-1)