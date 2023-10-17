from mindquantum import X, MQLayer, general_ghz_state
from mindquantum import Circuit, add_prefix, Hamiltonian, QubitOperator
from mindquantum import Simulator, MQAnsatzOnlyLayer, MQLayer, amplitude_encoder
from mindspore import nn
import mindspore.numpy as np
import mindspore as ms

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def elementary_layer(prefix="", num_qubits=2):
    """
    生成基本层，在QGAN中会用到
    """
    layer = Circuit()
    for i in range(num_qubits):
        layer.rx(f"rx-{i}", i)
        layer.ry(f"ry-{i}", i)
        layer.rz(f"rz-{i}", i)
    layer.un(X, [(i + 1) % num_qubits for i in range(num_qubits)],
             [i for i in range(num_qubits)])

    return add_prefix(layer, prefix)


def discriminator_circuit(num_qubits=2, layers=2):
    """
    生成D的线路
    """
    d = Circuit()
    for i in range(layers):
        d += elementary_layer(f"d-l{i}", num_qubits=num_qubits)
    return d


def gene_amplitude_encoder(num_qubits=2):
    """
    生成encoder量子线路
    用于num qubits个量子比特的amplitude encoder
    """
    data = [i for i in range(2**num_qubits)]
    encoder, _ = amplitude_encoder(data, n_qubits=num_qubits)
    return encoder


def generator_qubits_circuit(num_qubits=2, layers=2):
    """
    生成qubit circuit generater的线路
    """
    g = Circuit()
    for i in range(layers):
        g += elementary_layer(f"g-l{i}", num_qubits=num_qubits)
    return g


def generator_photonic_circuit(num_qubits=2):
    """
    生成photonic circuit generater的线路
    用到的wire是和基态空间个数一样的
    用到的知识是光量子计算的，是多能级的系统，
    """
    g = Circuit()
    for i in range(2**num_qubits):
        g.rx(f"g-rx-{i}", i)
        g.ry(f"g-ry-{i}", i)
        g.rz(f"g-rz-{i}", i)
    return g


def Generator(backbone):
    """
    实现photonic-like quantum generator
    输入是generator的量子线路，返回模型
    这个线路不需要输入，直接可以输出结果，输出是“测量后的光子数被映射到0，2”，确定了hamiltonian
    写成这样的形式，每次只能生成一个数据，shape是4
    """
    num_qubits = backbone.n_qubits
    hams = [
        Hamiltonian(QubitOperator(f"Z{i}") + QubitOperator(""))
        for i in range(num_qubits)
    ]
    simulator = Simulator("mqvector", n_qubits=num_qubits)
    grad_ops = simulator.get_expectation_with_grad(hams=hams,
                                                   circ_right=backbone)
    Generator = MQAnsatzOnlyLayer(grad_ops)
    return Generator


def Discriminator(encoder, ansatz):
    """
    实现带编码的D
    输入encoder，和D的circuit，返回模型
    模型输入encoder中对应的参数，对第一个量子比特测量，返回的值在-1， 1之间
    """
    num_qubits = encoder.n_qubits
    circuit = encoder.as_encoder() + ansatz.as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    simulator = Simulator("mqvector", n_qubits=num_qubits)
    grad_ops = simulator.get_expectation_with_grad(
        hams=ham,
        circ_right=circuit)
    Discriminator = MQLayer(grad_ops)
    return Discriminator


def vector_to_angle(data):
    """
    把generator生成的数据转为角度，因为在训练generator里面要用到，所以应该操作都用到ms里面的算子
    输入data，shape[d]，输出data对应的角度shape[1, d-1]
    现在直接实现4-->3 的转化
    """
    sum2 = np.sqrt(data[2] * data[2] + data[3] * data[3])
    sum1 = np.sqrt(data[0] * data[0] + data[1] * data[1])
    sum0 = np.sqrt(sum1 * sum1 + sum2 * sum2)
    angle = np.zeros([1, 3]).astype("float32")
    if sum0 > 1e-10:
        angle[0, 0] = 2 * np.arccos(sum1 / sum0)
    if sum1 > 1e-10:
        angle[0, 1] = 2 * np.arccos(data[0] / sum1)
    if sum2 > 1e-10:
        angle[0, 2] = 2 * np.arccos(data[2] / sum2)
    return angle


class GenewithDiscrim(nn.Cell):

    def __init__(self, G, convert, D):
        super(GenewithDiscrim, self).__init__()
        self.G = G
        self.convert = convert
        self.D = D

    def construct(self):
        x = self.G()
        x = self.convert(x)
        x = self.D(x)
        return x


if __name__ == "__main__":
    n_qubit = 2
    encoder = gene_amplitude_encoder(n_qubit)
    print(encoder)
