from mindquantum import UN, U3, X, CNOT, RY, RX, RZ
from mindquantum.algorithm import HardwareEfficientAnsatz                                           # 导入HardwareEfficientAnsatz
from mindquantum import add_prefix, Circuit, Hamiltonian, QubitOperator
from scipy.optimize import minimize
import numpy as np

import mindspore as ms                                                                         # 导入mindspore库并简写为ms
from mindquantum.framework import MQLayer                                                      # 导入MQLayer
from mindquantum.simulator import Simulator
import tqdm

from encoder_circuit import generate_encoder


ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)   


def generate_ansatz(n_qubits=3, depth=9):
    """生成ansatz线路
    """
    ansatz = HardwareEfficientAnsatz(n_qubits, single_rot_gate_seq=[RX, RY, RZ], entangle_gate=X, depth=depth).circuit  
    return ansatz, ansatz.params_name


def main():
    """主函数"""
    n_qubits = 3
    ## 1. 生成线路
    encoder, encoder_name = generate_encoder() # 编码器
    ansatz, ansatz_name = generate_ansatz() # 训练线路
    circuit = encoder + ansatz  # 合成的线路
    
    print("ansatz summary:\n")
    ansatz.summary()

    ## 2. 加载数据
    train_x = np.load('train_x.npy', allow_pickle=True)
    train_y = np.load('train_y.npy', allow_pickle=True)
    test_x = np.load('test_x.npy', allow_pickle=True)

    ## 3. 准备优化器等
    sim = Simulator('projectq', n_qubits)
    sim_left = Simulator('projectq', n_qubits)
    circuit_left = Circuit()
    ham = Hamiltonian(QubitOperator("")) # 哈密顿量

    ## 4. 梯度下降法训练
    lr = 0.01    # 学习率
    epoch = 50 
    ansatz_amp = np.array([0]*len(ansatz_name), dtype=np.float64)

    sim.reset()
    for i in range(1,epoch+1):
        facc = 0.0
        for x, y in zip(train_x, train_y):
            sim_left.set_qs(y)
            grad_ops = sim.get_expectation_with_grad(
                ham, 
                circ_right=circuit, 
                circ_left=Circuit(),
                simulator_left=sim_left,
                encoder_params_name=encoder_name,
                ansatz_params_name=ansatz_name,
                parallel_worker=4
            )
            f, g1, g2 = grad_ops(np.array([x]), ansatz_amp) # 计算梯度
            g2 = np.squeeze(g2)
            ansatz_amp = ansatz_amp + lr * g2.real  # 利用梯度更新权值
            facc += f[0,0].real
        if i % 10 == 0:
            print(f"epoch {i}, facc = {facc/len(train_x)}")
    print("Train accomplished!\n")

    ### 5. 测试在训练集的保真度
    fidelity = 0.0
    for xx, yy in zip(train_x, train_y):
        sim.reset()
        sim.apply_circuit(circuit, np.r_[xx, ansatz_amp])
        yout = sim.get_qs()
        fidelity += np.vdot(yout, yy).real
    fidelity /= len(train_x)
    print(f"Fidility on train dataset: {fidelity}")

    ### 6. 保存数据
    test_y = np.zeros(shape=(len(test_x), 8), dtype=train_y.dtype)
    for i, xx in enumerate(test_x):
        sim.reset()
        sim.apply_circuit(circuit, np.r_[xx, ansatz_amp]) 
        yout = sim.get_qs()
        test_y[i] = yout
    np.save("test_y.npy", test_y)
    print(f"Generate test_y successfully!")


if __name__ == '__main__':
    main()
