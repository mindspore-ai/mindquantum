import numpy as np
from mindquantum import Circuit, add_prefix, H
from mindquantum import Hamiltonian, QubitOperator, Simulator, MQLayer
from mindspore import nn
import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def frqi_encoder(num_pixels=64, compress=False):
    num_pos_qubit = int(np.log2(num_pixels))
    if compress:
        num_pos_qubit = num_pos_qubit - 2
    encoder = Circuit()
    encoder.un(H, list(range(num_pos_qubit)))
    for t, pos in enumerate(range(2**num_pos_qubit)):
        for k, c in enumerate('{0:0b}'.format(pos).zfill(num_pos_qubit)):
            if c == "0":
                encoder.x(k)
        encoder.ry({f'ry{t}': np.pi}, num_pos_qubit, [i for i in range(num_pos_qubit)])
        for k, c in enumerate('{0:0b}'.format(pos).zfill(num_pos_qubit)):
            if c == "0":
                encoder.x(k)
    return encoder


def CRADL(num_qubits=6, layers=1):
    """
    共有 num_qubits + 1 + 1个 qubits
    num_qubits: 存储位置
    color_qubit: 存储颜色
    readout_qubit: 读取结果    
    """
    ansatz = Circuit()
    for l in range(layers//2):
        for k in range(num_qubits):
            ansatz.xx(f"{l}-{k}xx", [k, num_qubits])
            ansatz.xx(f"{l}-{k}xx", [k, num_qubits+1])
        for k in range(num_qubits):
            ansatz.zz(f"{l}-{k}zz", [k, num_qubits])
            ansatz.zz(f"{l}-{k}zz", [k, num_qubits+1])
    return ansatz
    

def CRAML(num_qubits=6, layers=1):
    """
    共有 num_qubits + 1 + 1个 qubits
    num_qubits: 存储位置
    color_qubit: 存储颜色
    readout_qubit: 读取结果    
    """
    ansatz = Circuit()
    for l in range(layers):
        for k in range(num_qubits//2):
            ansatz.xx(f"{l}-{k}xx", [k, num_qubits])
            ansatz.xx(f"{l}-{k}xx", [k, num_qubits+1])
            ansatz.zz(f"{l}-{k}zz", [k, num_qubits])
            ansatz.zz(f"{l}-{k}zz", [k, num_qubits+1])
    return ansatz


class QNN(nn.Cell):
    
    def __init__(self, resolution, compress, num_layers, backend="projectq"):
        """
        图像大小，是否压缩，ansatz层数
        """
        super(QNN, self).__init__()
        # 创建编码线路
        num_pixels = resolution**2

        self.encoder = frqi_encoder(num_pixels, compress)
        self.encoder.as_encoder()

        # 创建ansatz线路
        num_qubits = int(np.log2(num_pixels))
        if compress:
            num_qubits -= 2
        self.ansatz = CRADL(num_qubits, num_layers)
        self.ansatz.as_ansatz()

        circuit = self.encoder + self.ansatz
        # 对高位执行测量
        self.ham = Hamiltonian(QubitOperator(f"Z{num_qubits+1}"))

        simulator = Simulator(backend, n_qubits=num_qubits+2)
        # 构建算期望和有关参数梯度的算子
        grad_ops = simulator.get_expectation_with_grad(hams=self.ham, circ_right=circuit)

        self.qnn = MQLayer(grad_ops)

    def construct(self, x):
        y = self.qnn(x)
        return y
