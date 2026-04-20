
import numpy as np                                         
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ,CNOT
from mindquantum.core.circuit import Circuit 
import math
from mindquantum.core.operators import QubitOperator          
from mindquantum.core.operators import Hamiltonian 
from mindquantum.core.parameterresolver import ParameterResolver as PR
import argparse
import pickle
def main(xargs):
    print(xargs)
    def FBS(a,b,theta,phi):
        FBS=Circuit()
        FBS+=X.on(a,b)
        FBS+=RZ(-math.pi-phi).on(b)
        FBS+=RY(-math.pi/2-theta).on(b)
        FBS+=X.on(b,a)
        FBS+=RY(math.pi/2+theta).on(b)
        FBS+=RZ(math.pi+phi).on(b)
        FBS+=X.on(a,b)
        return FBS

    import openfermion
    
    ham=openfermion.utils.load_operator(xargs.Hamiltonian_path,data_directory='')
    from mindquantum.core.operators.qubit_operator import QubitOperator
    from mindquantum.core.operators import Hamiltonian
    from mindquantum.simulator import Simulator

    ham=Hamiltonian(QubitOperator.from_openfermion(ham))
    
    # prepare_seed(xargs.Randseed)
    HF_circuit=Circuit([X.on(i) for i in range(xargs.K_num)]) 
    print(HF_circuit)
    n_params = len(HF_circuit.params_name)
    p0 = np.zeros(n_params)
    sim = Simulator('mqvector', xargs.Qubits_num)
    molecule_pqc = sim.get_expectation_with_grad(ham, HF_circuit)
    print(molecule_pqc(p0))
    exit(0)



if __name__=='__main__':
    parser = argparse.ArgumentParser("HWP")
    parser.add_argument("--Circuit_type", type=str, default = "fullconnect")# fullconnect HAE
    parser.add_argument("--Circuit_layers", type=int, default =5)
    parser.add_argument("--Gate_type", type=str, default="FBS")# RBS FBS
    parser.add_argument("--Hamiltonian_path", type=str,required=True)
    parser.add_argument("--Iteration", type=int, default=10000)
    parser.add_argument("--Learningrate", type=float, default=1e-3)
    parser.add_argument("--Lowest_energy", type=float)
    parser.add_argument("--Randseed", type=int,default=42)
    parser.add_argument("--Qubits_num", type=int,default=4)
    parser.add_argument("--K_num", type=int,default=2)
    parser.add_argument("--Output_path", type=str, required=True) # if enabled, the search space will be changed to Rx, Ry, Rz
    args = parser.parse_args()
    main(xargs=args)