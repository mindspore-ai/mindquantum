import numpy as np
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ, CNOT
from mindquantum.core.circuit import Circuit
import math
from mindquantum.core.operators import QubitOperator
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.core.gates import gene_univ_parameterized_gate
import argparse
import pickle
from mindquantum.simulator import Simulator, fidelity
import os

def prepare_seed(rand_seed):
    np.random.seed(rand_seed)
    # torch.manual_seed(rand_seed)
    # torch.cuda.manual_seed(rand_seed)
    # torch.cuda.manual_seed_all(rand_seed)


def main(xargs):



    best_state_info = {'best_energy': None, 'best_state': None}

    print(xargs)

    def FBS(a, b, theta, phi):
        FBS = Circuit()
        FBS += X.on(a, b)
        FBS += RZ(-math.pi - phi).on(b)
        FBS += RY(-math.pi / 2 - theta).on(b)
        FBS += X.on(b, a)
        FBS += RY(math.pi / 2 + theta).on(b)
        FBS += RZ(math.pi + phi).on(b)
        FBS += X.on(a, b)
        return FBS

    def matrix(alpha):
        ept = (np.exp(1j * alpha) - 1)
        return np.array([
            [1.0 + 0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, (ept + 2.0) / 2.0, (1.0 + 1.0j) / math.sqrt(8) * ept, 0.0j],
            [0.0j, (1.0 - 1.0j) / math.sqrt(8) * ept, (ept + 2.0) / 2.0, 0.0j],
            [0.0j, 0.0j, 0.0j, 1.0 + 0.0j],
        ])

    def diff_matrix(alpha):
        ep = (-np.sin(alpha) + np.cos(alpha) * 1j) / 2
        em = (-np.sin(alpha) + np.cos(alpha) * 1j)
        return np.array([
            [0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, ep, (1.0 + 1.0j) / math.sqrt(8) * em, 0.0j],
            [0.0j, (1.0 - 1.0j) / math.sqrt(8) * em, ep, 0.0j],
            [0.0j, 0.0j, 0.0j, 0.0j],
        ])

    def FanBS_matrix(alpha):
        ept = (np.exp(1j * alpha) - 1)
        return np.array([
            [1.0 + 0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, (ept + 2.0) / 2.0, (1.0 - 1.0j) / math.sqrt(8) * ept, 0.0j],
            [0.0j, (1.0 + 1.0j) / math.sqrt(8) * ept, (ept + 2.0) / 2.0, 0.0j],
            [0.0j, 0.0j, 0.0j, 1.0 + 0.0j],
        ])

    def FanBS_diff_matrix(alpha):
        ep = (-np.sin(alpha) + np.cos(alpha) * 1j) / 2
        em = (-np.sin(alpha) + np.cos(alpha) * 1j)
        return np.array([
            [0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, ep, (1.0 - 1.0j) / math.sqrt(8) * em, 0.0j],
            [0.0j, (1.0 + 1.0j) / math.sqrt(8) * em, ep, 0.0j],
            [0.0j, 0.0j, 0.0j, 0.0j],
        ])

    def RBS_matrix(alpha):
        COSalpha = math.cos(alpha)
        SINalpha = math.sin(alpha)
        return np.array([
            [1.0 + 0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, COSalpha, SINalpha, 0.0j],
            [0.0j, -SINalpha, COSalpha, 0.0j],
            [0.0j, 0.0j, 0.0j, 1.0 + 0.0j],
        ])

    def RBS_diff_matrix(alpha):
        COSalpha = math.cos(alpha)
        SINalpha = math.sin(alpha)
        return np.array([
            [0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, -SINalpha, COSalpha, 0.0j],
            [0.0j, -COSalpha, -SINalpha, 0.0j],
            [0.0j, 0.0j, 0.0j, 0.0j],
        ])

    BSGate = gene_univ_parameterized_gate('BS', matrix, diff_matrix)
    FanBSGate = gene_univ_parameterized_gate('FanBS', FanBS_matrix, FanBS_diff_matrix)
    RBSGate = gene_univ_parameterized_gate('RBS', RBS_matrix, RBS_diff_matrix)

    import openfermion
    prepare_seed(xargs.Randseed)
    circuit = Circuit()

    Theta = []
    Phi = []
    idx = 0
    circuit += X.on(0)
    for i in range(xargs.Qubits_num - xargs.K_num + 1, xargs.Qubits_num):
        circuit += X.on(i)
    # if xargs.Gate_type=='FBS':
    #     for _ in range(xargs.Circuit_layers):
    #         if xargs.Circuit_type=='fullconnect':
    #             for step in range(1,xargs.Qubits_num):
    #                 for i in range(0,xargs.Qubits_num-step):
    #                     Theta.append(PR('theta'+str(idx)))
    #                     Phi.append(PR('phi'+str(idx)))
    #                     circuit+=FBS(i,i+step,Theta[idx],Phi[idx])
    #                     idx+=1
    #         elif xargs.Circuit_type=='hea':
    #             for i in range(xargs.Qubits_num):
    #                 Theta.append(PR('theta'+str(idx)))
    #                 Phi.append(PR('phi'+str(idx)))
    #                 circuit+=FBS(i,(i+1)%xargs.Qubits_num,Theta[idx],Phi[idx])
    #                 idx+=1
    if xargs.Gate_type == 'RBS':
        if xargs.Circuit_type == 'FC':
            for _ in range(xargs.Circuit_layers):
                for step in range(1, xargs.Qubits_num):
                    for i in range(0, xargs.Qubits_num - step):
                        Theta.append(PR('theta' + str(idx)))
                        circuit += RBSGate(Theta[idx]).on([i, i + step])
                        idx += 1
        elif xargs.Circuit_type == 'newhea':
            for layers in range(xargs.Circuit_layers):
                for i in range(0, xargs.Qubits_num-1, 2):
                    Theta.append(PR('theta' + str(idx)))
                    circuit += RBSGate(Theta[idx]).on([i, (i + 1) % xargs.Qubits_num])
                    idx += 1
                for i in range(1, xargs.Qubits_num-1, 2):
                    Theta.append(PR('theta' + str(idx)))
                    circuit += RBSGate(Theta[idx]).on([i, (i + 1) % xargs.Qubits_num])
                    idx += 1
    elif xargs.Gate_type == 'BS':
        if xargs.Circuit_type == 'FC':
            for _ in range(xargs.Circuit_layers):
                for step in range(1, xargs.Qubits_num):
                    for i in range(0, xargs.Qubits_num - step):
                        Theta.append(PR('theta' + str(idx)))
                        circuit += BSGate(Theta[idx]).on([i, i + step])
                        idx += 1
        elif xargs.Circuit_type == 'newhea':
            for layers in range(xargs.Circuit_layers):
                for i in range(0,xargs.Qubits_num-1,2):
                    Theta.append(PR('theta'+str(idx)))
                    if layers%2==0:
                        circuit+=BSGate(Theta[idx]).on([i,(i+1)%xargs.Qubits_num])
                    else:
                        circuit+=FanBSGate(Theta[idx]).on([i,(i+1)%xargs.Qubits_num])
                    idx+=1        
                for i in range(1,xargs.Qubits_num-1,2):
                    Theta.append(PR('theta'+str(idx)))
                    if layers%2==0:
                        circuit+=BSGate(Theta[idx]).on([i,(i+1)%xargs.Qubits_num])
                    else:
                        circuit+=FanBSGate(Theta[idx]).on([i,(i+1)%xargs.Qubits_num])
                    idx+=1        

    elif xargs.Gate_type == 'RCNOT':
        for _ in range(xargs.Circuit_layers):
            for i in range(xargs.Qubits_num):
                Theta.append(PR('theta' + str(idx)))
                Phi.append(PR('phi' + str(idx)))
                circuit += RY(Theta[idx]).on(i)
                circuit += RZ(Phi[idx]).on(i)
                idx += 1
            for i in range(xargs.Qubits_num):
                circuit += X.on((i + 1) % xargs.Qubits_num, i)

    # print(circuit)
    ham = openfermion.utils.load_operator(xargs.Hamiltonian_path, data_directory='')
    from mindquantum.core.operators.qubit_operator import QubitOperator
    from mindquantum.core.operators import Hamiltonian
    from mindquantum.simulator import Simulator

    ham = Hamiltonian(QubitOperator.from_openfermion(ham))

    # print(ham)

    # sim = Simulator('mqvector', circuit.n_qubits)
    sim = Simulator('mqvector_gpu', circuit.n_qubits)
    molecule_pqc = sim.get_expectation_with_grad(ham, circuit)
    n_params = len(circuit.params_name)
    p0 = np.random.rand(n_params)

    sim_copy = Simulator('mqvector_gpu', circuit.n_qubits)  # 创建模拟器副本
    circuit_copy = circuit


    # def fun(p0, molecule_pqc, timestart, energy_list=None):
    #     f, g = molecule_pqc(p0)
    #     f = np.real(f)[0, 0]
    #     g = np.real(g)[0, 0]
    #     if energy_list is not None:
    #         energy_list.append(f)
    #         if len(energy_list) % 1 == 0:
    #             print(f"Step: {len(energy_list)},\tenergy: {f},\t time:{time.time() - timestart}s")
    #     return f, g
    def fun(p0, molecule_pqc, sim, circuit, fci_state, best_state_info, timestart, energy_list=None):
    # 计算期望值和梯度
        f, g = molecule_pqc(p0)
        f = np.real(f)[0, 0]
        g = np.real(g)[0, 0]

        param_resolver = PR({name: value for name, value in zip(circuit.params_name, p0)})
        # print(p0)
        # 重置模拟器状态并应用当前的参数化电路
        sim_copy.reset()
        sim_copy.apply_circuit(circuit_copy, param_resolver)


        current_state = sim_copy.get_qs()


        # 计算与FCI状态的保真度
        current_fidelity = fidelity(current_state, fci_state)

        # 仅当找到更低的能量时保存状态
        if best_state_info['best_energy'] is None or f < best_state_info['best_energy']:
            best_state_info['best_energy'] = f
            best_state_info['best_state'] = current_state
            best_state_path = os.path.join('/mnt/nas-new/home/yange/nature_comm/pack3/fidelity', 'best_state.pkl')
            with open(best_state_path, 'wb') as f_out:
                pickle.dump(best_state_info['best_state'], f_out)

        # 打印优化过程中的信息
        if energy_list is not None:
            energy_list.append(f)
            print(f"Step: {len(energy_list)},\tenergy: {f},\tfidelity: {current_fidelity},\ttime: {time.time() - timestart}s")

        # 返回值用于优化：能量和梯度
        return f, g




        


    from scipy.optimize import minimize
    import time
    timestart = time.time()


    fci_state_path = os.path.join('/mnt/nas-new/home/yange/nature_comm/pack3/fidelity', 'H2_fci_state.pkl')
    with open(fci_state_path, 'rb') as f:
        fci_state = pickle.load(f)



    energy_list = []
    # res = minimize(fun, p0, args=(molecule_pqc, timestart, energy_list), method='bfgs', jac=True, tol=0,
    #                options={'maxiter': xargs.Iteration, 'gtol': 0})
    res = minimize(fun, p0, args=(molecule_pqc, sim, circuit, fci_state, best_state_info, timestart, energy_list),
                method='bfgs', jac=True, tol=0, options={'maxiter': xargs.Iteration, 'gtol': 0})

    print(res)
    final_state1 = sim_copy.get_qs()
    final_state2 = sim.get_qs()
    print(f"final_state1:{final_state1}")
    # print(f"final_state2:{final_state2}")
    print(f"fci_state:{fci_state}")
    final_fidelity1=fidelity(final_state1, fci_state)
    # final_fidelity2=fidelity(final_state2, fci_state)
    print(f"final_fidelity1:{final_fidelity1}")
    # print(f"final_fidelity2:{final_fidelity2}")


    Lowest_energy = pickle.load(open(xargs.Lowest_energy_path, "rb"))
    A = [xargs, None, p0, res, energy_list]
    # print(A)
    with open(xargs.Output_path, "wb") as f:
        pickle.dump(A, f)

    print(np.min(energy_list) - (Lowest_energy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("HWP")
    parser.add_argument("--Circuit_type", type=str, default="fullconnect")  # fullconnect HAE
    parser.add_argument("--Circuit_layers", type=int, default=5)
    parser.add_argument("--Gate_type", type=str, default="FBS")  # RBS FBS
    parser.add_argument("--Hamiltonian_path", type=str, required=True)
    parser.add_argument("--Iteration", type=int, default=10000)
    parser.add_argument("--Learningrate", type=float, default=1e-3)
    parser.add_argument("--Lowest_energy_path", type=str)
    parser.add_argument("--Randseed", type=int, default=42)
    parser.add_argument("--Qubits_num", type=int, default=4)
    parser.add_argument("--K_num", type=int, default=2)
    parser.add_argument("--Output_path", type=str,
                        required=True)  # if enabled, the search space will be changed to Rx, Ry, Rz
    args = parser.parse_args()
    main(xargs=args)