import os
import sys

sys.path.append(os.path.abspath(__file__))
from simulator import HKSSimulator
import simulator
from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data
from mindquantum.core.operators import QubitOperator, Hamiltonian, TimeEvolution, commutator
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import X,RZ, RY, CNOT
from mindquantum import uccsd_singlet_generator, SingleLoopProgress
from mindquantum.core.parameterresolver import ParameterResolver

import numpy as np
from scipy.optimize import minimize
from mindquantum.simulator import Simulator
# from mindquantum.algorithm.nisq import Transform
# from MPS import train_mps_basic_state, mps2circparams



"""Error mitigation algorithm."""
from mindquantum.algorithm.error_mitigation import fold_at_random
from mindspore import nn, Tensor, dtype as mstype
from mindspore import context
import torch as tc
import datetime

import matplotlib.pyplot as plt

context.set_context(mode=context.GRAPH_MODE)

class ZNENet(nn.Cell):
    def __init__(self):
        super(ZNENet, self).__init__()
        self.fc1 = nn.Dense(1, 20, activation='relu')
        self.fc2 = nn.Dense(20, 10, activation='relu')
        self.fc3 = nn.Dense(10, 1)


    def construct(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def train_and_extrapolate(scale_factors, expectations, epochs=10000, initial_lr=0.01, end_lr=0.01, power=1.0):
    # 创建模型
    model = ZNENet()

    # 损失函数和优化器
    criterion = nn.L1Loss()
    total_steps = epochs
    lr_schedule = nn.PolynomialDecayLR(initial_lr, end_lr, total_steps, power=power)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr_schedule)

    # 训练模型
    train_network = nn.WithLossCell(model, criterion)
    train_network = nn.TrainOneStepCell(train_network, optimizer)

    # 训练循环
    for epoch in range(epochs):
        loss = train_network(Tensor(scale_factors, dtype=mstype.float32), Tensor(expectations, dtype=mstype.float32))
        if epoch % 1000 == 999:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.asnumpy():.6f}')

    # 外推到零噪声
    zero_noise = Tensor(np.array([[0.0]], dtype=np.float32), dtype=mstype.float32)
    model.set_train(False)  # 设置模型为评估模式
    predicted_zero_noise = model(zero_noise)
    return predicted_zero_noise.asnumpy()[0][0]


def zne(
    circuit: Circuit,
    params: np.ndarray,
    Simulator: HKSSimulator,
    split_ham,
    shots: int,
    scaling=None,
    order: float = 1.0,
    method="R",
    a=0,
    args=None,
    Z_hams_is=True,
    n_sampling=3,
expec_for_scaling1=None
) -> float:
    """
    Zero noise extrapolation.

    Args:
        circuit (:class:`~.core.circuit.Circuit`): A quantum circuit.
        params: np.ndarray, The parameters of the circuit.
        Simulator: The simulator to use. It should be a subclass of :class:`~.simulator.Simulator`.
        split_ham (List[QubitOperator]): The split Hamiltonian except the constant term.
        const (float): The constant term of the Hamiltonian.
        shots (int): The number of sampling shots.
        scaling (List[float]): The scaling factor to folding circuit. If ``None``, it will be ``[1.0, 2.0, 3.0]``.
            Default: ``None``.
        order (float): Order of extrapolation for polynomial. Default: ``None``.
        method (str): Extrapolation method, could be ``'R'`` (Richardson), ``'P'`` (polynomial) and
            ``'PE``' (poly exponential). Default: ``'R'``.
        a (float): Poly exponential extrapolation factor. Default: ``0``.
        args (Tuple): The other arguments for executor except first one.
    """
    y = []
    mitigated = 0
    print(f"zne with method: {method}\n")
    if scaling is None:
        scaling = [1.0, 2.0, 3.0]

    min_expectations = []
    for repetition in range(5):
        scaled_circuit = fold_at_random(circuit, scaling[0])

        split_ham_expectation_i_list = []
        for sampling_i in range(n_sampling * 2):
            split_ham_expectation_i = mea_all_ham(scaled_circuit, params, split_ham, Simulator, shots, Z_hams_is)
            split_ham_expectation_i_list.append(split_ham_expectation_i)

        current_min_expectation = min(split_ham_expectation_i_list)
        min_expectations.append(current_min_expectation)
        print(f"Repetition {repetition + 1}, split_hams Expectation for factor {scaling[0]}: {current_min_expectation}")


    final_min_expectation = min(min_expectations)
    print(f"Finding minimum expectation for factor {scaling[0]}: {final_min_expectation}")
    if expec_for_scaling1 is not None:
        final_min_expectation = min(final_min_expectation, expec_for_scaling1)
    y.append(final_min_expectation)
    print(f"Final minimum expectation for factor {scaling[0]}: {final_min_expectation}")

    previous_expectation = final_min_expectation
    for factor in scaling[1:]:
        expecs_for_factor = []
        for attempt in range(5):
            scaled_circuit = fold_at_random(circuit, factor)
            split_ham_expectation = 0
            for sampling_i in range(n_sampling):
                split_ham_expectation_i = mea_all_ham(scaled_circuit, params, split_ham, Simulator, shots, Z_hams_is)
                split_ham_expectation += split_ham_expectation_i
            split_ham_expectation /= n_sampling
            print(f"Attempt {attempt + 1} - split_hams Expectation for factor {factor}: {split_ham_expectation}")
            expecs_for_factor.append(split_ham_expectation)

            # Only append if it's greater than the previous expectation
            if split_ham_expectation > previous_expectation:
                y.append(split_ham_expectation)
                previous_expectation = split_ham_expectation
                break
            elif attempt == 4:
                print(f"Failed to find a larger expectation after 5 attempts for factor {factor}.")
                # Optionally, append the last attempt or handle it differently
                y.append(max(expecs_for_factor))
                previous_expectation = max(expecs_for_factor)


    if method == "R":
        for k, y_k in enumerate(y):
            product = 1
            for i in range(0, len(y)):
                if k != i:
                    try:
                        product = product * (scaling[i] / (scaling[i] - scaling[k]))
                    except ZeroDivisionError as exc:
                        raise ZeroDivisionError(f"Error scaling: {scaling}") from exc
            mitigated = mitigated + y_k * product
        return mitigated
    if order is None:
        raise ValueError("For polynomial and poly exponential, order cannot be None.")
    if method == "P":
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
        return mitigated
    if method == "PE":
        y = y - a
        y = np.log(y)
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
    if method == "ML":
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        scale_factors = Tensor(np.array([[np.array(scaling_i)] for scaling_i in scaling]))
        split_ham_expectations = Tensor(np.array([[np.array(y_i)] for y_i in y]))
        mitigated = train_and_extrapolate(scale_factors.astype(np.float32), split_ham_expectations.astype(np.float32))
        print(f'Extrapolated zero-noise expectation: {mitigated:.4f} with machine learning')
    else:
        print("Provide a valid extrapolation scheme R, PE, P, ML")

    return mitigated


def get_mps_circuit(num_qubits, n_layers):
    """
    创建一个MPS量子电路。
    参数:
        num_qubits (int): 量子比特的数量。
    返回:
        circuit (Circuit): 构建的量子电路。
    """
    circuit = Circuit()

    def add_block(circ, qubits, idx):
        """在一对量子比特上添加RZ, RY, RZ门和一个CNOT门，每个门使用不同的参数。"""
        q1, q2 = qubits
        circ += RZ(f'alpha_{idx}_L1').on(q1)
        circ += RY(f'beta_{idx}_L1').on(q1)
        circ += RZ(f'gamma_{idx}_L1').on(q1)
        circ += RZ(f'alpha_{idx}_L2').on(q2)
        circ += RY(f'beta_{idx}_L2').on(q2)
        circ += RZ(f'gamma_{idx}_L2').on(q2)
        circ += CNOT.on(q1, q2)
        circ += RZ(f'alpha_{idx}_R1').on(q1)
        circ += RY(f'beta_{idx}_R1').on(q1)
        circ += RZ(f'gamma_{idx}_R1').on(q1)
        circ += RZ(f'alpha_{idx}_R2').on(q2)
        circ += RY(f'beta_{idx}_R2').on(q2)
        circ += RZ(f'gamma_{idx}_R2').on(q2)

    for layer in range(n_layers):
        for i in range(num_qubits - 2):
            add_block(circuit, [i, i+1], i + layer*n_layers)
        circuit += RZ(f'alpha_z1_last').on(num_qubits - 1)
        circuit += RY(f'beta_y_last').on(num_qubits - 1)
        circuit += RZ(f'gamma_z2_last').on(num_qubits - 1)

    return circuit

def split_hamiltonian(ham: QubitOperator):
    const = 0
    split_ham = []
    for i, j in ham.split():
        if j == 1:
            const = i.const.real
        else:
            split_ham.append([i.const.real, j])
    return const, split_ham


def rotate_to_z_axis_and_add_measure(circ: Circuit, ops: QubitOperator):
    circ = circ.copy()
    assert ops.is_singlet
    for idx, o in list(ops.terms.keys())[0]:
        if o == 'X':
            circ.ry(-np.pi / 2, idx)
        elif o == 'Y':
            circ.rx(np.pi / 2, idx)
        circ.measure(idx)
    return circ



def get_best_mps_params(ham, mps_circ, p0=None, output_file='f_history.txt'):
    circ = mps_circ

    if p0 == None:
        p0 = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
    else:
        p0 = tc.cat(p0).detach().numpy()

    grad_ops = Simulator('mqvector', circ.n_qubits).get_expectation_with_grad(
        Hamiltonian(ham), circ)

    def fun(x, grad_ops):
        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g

    f_history = []

    def callback(xk):
        f, g = fun(xk, grad_ops)
        f_history.append(f)

    res = minimize(fun, p0, (grad_ops,), 'bfgs', True, callback=callback)

    # 保存损失变化到文件
    with open(output_file, "w") as f:
        for loss in f_history:
            f.write(str(loss) + "\n")

    # 绘制损失图
    plt.figure(figsize=(12, 6))
    plt.plot(f_history)
    plt.xlabel("Iteration")
    plt.ylabel("Fun_value")
    plt.title("Optimization Fun_value History")
    plt.show()

    return res.x


def get_optimal_scaling_circit_params(circuit, split_ham, Simulator, shots=100, scaling=[1.0, 2.0, 3.0],
                                      Z_hams_is=True, ham=None, p0=None, mol_name='H4'):
    difer_params_circ_min_expectations = []
    params_list = []
    n_circuit = 1
    iters = 1
    n_sampling = 3

    for repetition in range(n_circuit):
        print(f"init circuit params for molecule {mol_name}: {repetition + 1}")
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if p0==None:
            params = get_best_mps_params(ham, circuit, output_file = f'energy_history_without_mps_init_{repetition}_{mol_name}_{current_time}.txt')
        else:
            params = get_best_mps_params(ham, circuit, p0=p0, output_file=f'energy_history_with_mps_init_{repetition}_{mol_name}_{current_time}.txt')

        params_list.append(params)

        current_params_circ_min_expectations = []
        for iter in range(iters):
            scaled_circuit = fold_at_random(circuit, scaling[0])
            split_ham_expectation_i_list = []
            for sampling_i in range(n_sampling):
                split_ham_expectation_i = mea_all_ham(scaled_circuit, params, split_ham, Simulator, shots, Z_hams_is)
                split_ham_expectation_i_list.append(split_ham_expectation_i)
            current_fold_circ_min_expectation = min(split_ham_expectation_i_list)
            current_params_circ_min_expectations.append(current_fold_circ_min_expectation)
        min_current_params_circ_expectations = min(current_params_circ_min_expectations)
        difer_params_circ_min_expectations.append(min_current_params_circ_expectations)
        print(
            f"Repetition {repetition + 1}, split_hams Expectation for factor {scaling[0]}: {min_current_params_circ_expectations}")

    final_min_expectation = min(difer_params_circ_min_expectations)
    min_expectation_index = difer_params_circ_min_expectations.index(final_min_expectation)
    optimal_params = params_list[min_expectation_index]
    print(f"Final minimum expectation for factor {scaling[0]}: {final_min_expectation}, the corespoding params index is {min_expectation_index}")
    return optimal_params, final_min_expectation



def mea_all_ham(circ, p, split_ham, Simulator: HKSSimulator, shots, Z_ham_is=True):
    result= 0
    if Z_ham_is:
        Z_hams, rest_hams = extract_Z_hamiltonians(split_ham)
        result += mea_all_Z_ham(circ, Z_hams, p, Simulator)
        with SingleLoopProgress(len(rest_hams), '非Z哈密顿量测量中') as bar:
            for idx, (coeff, ops) in enumerate(rest_hams):
                result += mea_single_ham(circ, ops, p, Simulator, shots) * coeff
                bar.update_loop(idx)
        return result
    else:
        with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
            for idx, (coeff, ops) in enumerate(split_ham):
                result += mea_single_ham(circ, ops, p, Simulator, shots) * coeff
                bar.update_loop(idx)
        return result

def mea_single_ham(circ, ops, p, Simulator: HKSSimulator, shots=100):
    circ = rotate_to_z_axis_and_add_measure(circ, ops)
    pr = ParameterResolver(dict(zip(circ.params_name, p)))
    sim = Simulator('mqvector', circ.n_qubits)
    result = sim.sampling(circ, shots=shots, pr=pr)
    expec = 0
    for i, j in result.data.items():
        expec += (-1)**i.count('1') * j / shots
    return expec

def mea_all_Z_ham(circ, Z_hams, p, Simulator: HKSSimulator, shots=100):
    circ = circ.copy()
    measure_qubits = set()
    Z_hams_qubit_index = []
    for coeff, ham in Z_hams:
        Z_ham_qubit_index = []
        for term, _ in ham.terms.items():
            if len(term) == 1:
                qubit_index, operator = term[0]
                measure_qubits.add(qubit_index)
                Z_ham_qubit_index.append(qubit_index)
            else:
                for qubit_index, operator in term:
                    measure_qubits.add(qubit_index)
                    Z_ham_qubit_index.append(qubit_index)
        Z_hams_qubit_index.append(Z_ham_qubit_index)

    for measure_qubit in measure_qubits:
        circ.measure(measure_qubit)

    pr = ParameterResolver(dict(zip(circ.params_name, p)))
    sim = Simulator('mqvector', circ.n_qubits)
    result = sim.sampling(circ, shots=shots, pr=pr)
    expec = 0
    for coeffi_index, ham_indices in enumerate(Z_hams_qubit_index):
        expec_ham_i = 0
        for i, j in result.data.items():
            # 计算每个片段中'1'的数量
            # count = sum(i[x] == '1' for x in ham_indices)
            count = sum(i[len(i)-1-x] == '1' for x in ham_indices)
            expec_ham_i += (-1) ** count * j / shots
        expec += expec_ham_i * Z_hams[coeffi_index][0]
    return expec



def mea_group_ham(circ, group_ops, coeffs, p, Simulator: HKSSimulator, shots=100):

    circ = circ.copy()

    measure_qubits = []
    for coeff, ops in group_ops:
        for ham, _ in ops.terms.items():  # 假设ops是QubitOperator且包含一个项
            if len(ham) == 1:
                qubit_index, operator = ham[0]
                if operator == 'X':
                    circ.ry(-np.pi / 2, qubit_index)
                elif operator == 'Y':
                    circ.rx(np.pi / 2, qubit_index)
                measure_qubits.append(qubit_index)
                # circ.measure(qubit_index)
            else:
                for qubit_index, operator in ham:
                    if operator == 'X':
                        circ.ry(-np.pi / 2, qubit_index)
                    elif operator == 'Y':
                        circ.rx(np.pi / 2, qubit_index)
                    measure_qubits.append(qubit_index)
                    # circ.measure(qubit_index)

    measure_qubits = list(dict.fromkeys(measure_qubits))
    for qubit_index in measure_qubits:
        circ.measure(qubit_index)

    # 设置参数解析器
    pr = ParameterResolver(dict(zip(circ.params_name, p)))

    # 运行仿真器
    sim = Simulator('mqvector', circ.n_qubits)
    result = sim.sampling(circ, shots=shots, pr=pr)

    # 计算期望值
    expec = 0
    for i, j in result.data.items():
        expec += (-1) ** i.count('1') * j / shots

    total_expectation = sum(coeff * expec for coeff in coeffs)

    return total_expectation


def find_commuting_groups(hamiltonian):
    commuting_groups = {}
    for coeff, term in hamiltonian:
        inserted = False
        for key, group in commuting_groups.items():
            # 检查当前项与已有组中的项是否全部对易
            if all((commutator(term, g[1]) == 0 for g in group)):
                group.append((coeff, term))
                inserted = True
                break
        if not inserted:
            commuting_groups[f"group_{len(commuting_groups)}"] = [(coeff, term)]
    return commuting_groups


def extract_Z_hamiltonians(hamiltonian):
    Z_hams = []
    rest_hams = []
    for coeff, term in hamiltonian:
        for ham, _ in term.terms.items():
            rest_ham_flag = False
            # 假设term是QubitOperator且包含一个项
            if len(ham) == 1:
                qubit_index, operator = ham[0]
                if operator == 'Z':
                    Z_hams.append((coeff, term))
                else:
                    rest_hams.append((coeff, term))
            else:
                if all((operator == 'Z' for _, operator in ham)):
                    Z_hams.append((coeff, term))
                else:
                    rest_hams.append((coeff, term))

    return Z_hams, rest_hams


def solution(molecule, Simulator: simulator.HKSSimulator) -> float:
    group_Z_hams_is = True
    # group_Z_hams_is = False
    zne_is = True
    method = 'ML'
    n_layers = 1
    scaling = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    n_sampling = 1

    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)
    Z_hams, rest_hams = extract_Z_hamiltonians(split_ham)

    ucc = get_mps_circuit(mol.n_qubits + 1, n_layers=n_layers)
    ucc.summary()
    print('FCI_energy:', mol.fci_energy)

    # loss_list, energy_list, trained_mps_tensors = train_mps_basic_state(iter=500, H=ham.__str__(), target_energy=mol.fci_energy, mol_qubits=mol.n_qubits)
    # print("energy_list:", energy_list)
    # tc.save(trained_mps_tensors, 'trained_mps_tensors.pt')
    # pretrained_circ_params = mps2circparams(trained_mps_tensors, iter=50, filename='trained_mps_tensors.pt')
    # p, min_expec_for_scaling1 = get_optimal_scaling_circit_params(ucc, split_ham, Simulator,
    #                                                               shots=100, scaling=scaling, Z_hams_is=group_Z_hams_is,
    #                                                               ham=ham, p0=pretrained_circ_params, mol_name=mol.name)

    p, min_expec_for_scaling1 = get_optimal_scaling_circit_params(ucc, split_ham, Simulator,
                                                                  shots=100, scaling=scaling, Z_hams_is=group_Z_hams_is,
                                                                  ham=ham, mol_name=mol.name)

    if Simulator == HKSSimulator:
        print("======HKSSimulator======\n")
        if zne_is:
            result_with_zne_Z_hams = const
            result_with_zne_Z_hams += zne(ucc, p, Simulator, split_ham, shots=100, scaling=scaling, method=method, Z_hams_is=group_Z_hams_is, n_sampling=n_sampling, expec_for_scaling1=min_expec_for_scaling1)
            return result_with_zne_Z_hams
        else:
            if group_Z_hams_is:
                result_with_Z_hams = const
                result_with_Z_hams += mea_all_Z_ham(ucc, Z_hams, p, Simulator)
                with SingleLoopProgress(len(rest_hams), '非Z哈密顿量测量中') as bar:
                    for idx, (coeff, ops) in enumerate(rest_hams):
                        result_with_Z_hams += mea_single_ham(ucc, ops, p, Simulator) * coeff
                        bar.update_loop(idx)
                return result_with_Z_hams
            else:
                result = const
                with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
                    for idx, (coeff, ops) in enumerate(split_ham):
                        result += mea_single_ham(ucc, ops, p, Simulator) * coeff
                        bar.update_loop(idx)
                return result

    else:
        print("======Simulator======\n")
        result = const
        if group_Z_hams_is:
            result += mea_all_Z_ham(ucc, Z_hams, p, Simulator)
            with SingleLoopProgress(len(rest_hams), '非Z哈密顿量测量中') as bar:
                for idx, (coeff, ops) in enumerate(rest_hams):
                    result += mea_single_ham(ucc, ops, p, Simulator) * coeff
                    bar.update_loop(idx)
            result_with_Z_hams = result
            return result_with_Z_hams
        else:
            result = const
            with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
                for idx, (coeff, ops) in enumerate(split_ham):
                    result += mea_single_ham(ucc, ops, p, Simulator) * coeff
                    bar.update_loop(idx)
            return result




if __name__ == '__main__':
    import simulator
    simulator.init_shots_counter()

    for num in range(1):
        for data_path, mol_name in [('data_mol/mol_H4.csv', 'H4')]:
            molecule = read_mol_data(data_path)
            print(f"mol_name: {mol_name}")
            Ansatz = 'mps'
            # print(f"Ansatz: {Ansatz}")
            HKSSimulator_energy = []
            Simulator_energy = []
            for num in range(1):
                for sim in [HKSSimulator, Simulator]:
                    result = solution(molecule, sim)
                    print(sim, result)
                    if sim == HKSSimulator:
                        HKSSimulator_energy.append(result)
                    else:
                        Simulator_energy.append(result)
                print(f"finished {num}th for {Ansatz}")

            # save_path = 'Qubit_UCC_results'
            # save_path = 'HardwareEfficient_results'
            # save_path = 'UCCSD_results'
            # save_path = 'StronglyEntangling_results'
            # save_path = 'results/' + mol_name + '/' + Ansatz + '_results_' + str(num)
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # np.save(f'{save_path}/HKSSimulator_energy.npy', HKSSimulator_energy)
            # np.save(f'{save_path}/Simulator_energy.npy', Simulator_energy)
