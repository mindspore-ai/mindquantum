import numpy as np
import networkx as nx
from mindquantum import Circuit, H, ZZ, RX, BarrierGate
from mindquantum import Hamiltonian, QubitOperator, Simulator, MQEncoderOnlyOps
from mindspore import nn, ops, Tensor
import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def fig7_instance1():
    """
    生成论文中fig7 n=6 的图
    """
    g = nx.Graph()
    nx.add_path(g, [0, 1])
    nx.add_path(g, [0, 3])
    nx.add_path(g, [0, 4])
    nx.add_path(g, [0, 5])
    nx.add_path(g, [1, 3])
    nx.add_path(g, [1, 4])
    nx.add_path(g, [1, 5])

    return g


def fig7_instance2():

    g = nx.Graph()
    nx.add_path(g, [0, 2])
    nx.add_path(g, [0, 4])
    nx.add_path(g, [1, 2])
    nx.add_path(g, [1, 3])
    nx.add_path(g, [1, 4])
    nx.add_path(g, [1, 5])
    nx.add_path(g, [1, 6])
    nx.add_path(g, [1, 7])
    nx.add_path(g, [2, 5])
    nx.add_path(g, [2, 6])
    nx.add_path(g, [2, 7])
    nx.add_path(g, [3, 4])
    nx.add_path(g, [3, 6])
    nx.add_path(g, [4, 5])
    nx.add_path(g, [5, 7])
    return g


def gene_random_instance(num_nodes):
    """
    函数生成随机图
    输入是 图的结点个数 num_nodes, 文章里是n
    输出是图
    """
    k = np.random.randint(3, num_nodes)
    # 两个顶点连接的概率 p = k / n, 文中没说生成 k-regular graph.
    p = k / num_nodes
    g = nx.Graph()
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.random() < p:
                nx.add_path(g, [i, j])
    return g


def gene_Hc_circuit(G, param):
    """
    生成HC2对应的线路, 也就是ZZ门作用到对应的qubits上
    input: G maxcut的图, param 这一层Hc对应的参数
    output: HC2 circuit
    """
    hc_circuit = Circuit()
    for e in G.edges:
        hc_circuit += ZZ(param).on(e)
    hc_circuit += BarrierGate(False)
    return hc_circuit


def gene_Hm_circuit(G, param):
    """
    生成HM对应的线路, 也就是RX门作用到对应的qubits上
    input: G maxcut的图, param 这一层Hc对应的参数
    output: HM circuit
    """
    hm_circuit = Circuit()
    for v in G.nodes:
        hm_circuit += RX(param).on(v)
    hm_circuit += BarrierGate(False)
    return hm_circuit


def gene_qaoa_ansatz(G, P):
    """
    使用 hc hm 生成 qaoa的ansatz
    input: G, 对应的图  P: hc hm 重复的次数
    output: ansatz circuit
    """
    ansatz = Circuit()
    for i in G.nodes:
        ansatz += H(i)
    for i in range(P):
        ansatz += gene_Hc_circuit(G, f"g{i}")
        ansatz += gene_Hm_circuit(G, f"m{i}")
    return ansatz


def gene_ham(G):
    ham = QubitOperator()
    for i in G.edges:
        ham += QubitOperator(f'Z{i[0]} Z{i[1]}')  # 生成哈密顿量Hc
    return Hamiltonian(ham)


def gene_qaoa_layers(G, P):
    """
    使用mindquantum framework 生成 QNN的层, 输出是期望值, 因为不需要在这里面更新参数，所以使用MQEncoderOnlyOps
    """
    ansatz = gene_qaoa_ansatz(G, P)
    ansatz.as_encoder()

    ham = gene_ham(G)
    sim = Simulator('projectq', ansatz.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, ansatz)
    net = MQEncoderOnlyOps(grad_ops)
    return net


class MetaQAOA(nn.Cell):
    """
    生成 MetaQAOA网络, 把LSTM看成主体结构, QNN也包含在内
    准备把参数看成是[1, P, 2], P 是QAOA的层数, 2 代表每一层里面两个参数
    LSTM需要初始化input_size, hidden_size (int), num_layers (int), 这里设置成[2, 2, lstm_layers]
    h_dim = 2 是为了输出的size也是2, 如果要对应到论文中h=0一维的话, 可以bidirectional=True

    QNN需要外部输入
    """
    def __init__(self, T, QNN, lstm_layers):
        """
        T times of quantum classical interaction 
        """
        super(MetaQAOA, self).__init__()
        # 初始化LSTM三个参数为input_size, hidden_size (int), num_layers (int)
        self.lstm = nn.LSTM(2, 2, lstm_layers, has_bias=True, batch_first=True, bidirectional=False)
        self.T = T
        self.qnn = QNN

    def construct(self, theta, h, c): # 
        """
        output:
        1. theta, h  是最后一步的
        2. y_list [T, (1)] 用于计算loss        
        """
        # theta_list = []
        y_list = []
        for i in range(self.T):
            y = self.qnn(theta.reshape(1, -1))
            y_list.append(y.squeeze())
            y = y.reshape(1, 1, 1)
            theta, (h, c) = self.lstm(theta, (h, c))
            # theta_list.append(theta)
            
        return theta, h, c, y_list



class MetaQAOALoss(nn.LossBase):
    def __init__(self):
        super(MetaQAOALoss, self).__init__()
        self.addn = ops.AddN()

    def construct(self, y_list):
        length = len(y_list)
        min_temp = y_list[0]
        loss_list = [min_temp]
        # loss_list = [min_temp]
        for i in range(1, length):
            loss_t = y_list[i] - min_temp
            if loss_t < 0:
                loss_list.append(loss_t)
            else:
                loss_list.append(Tensor(np.zeros([], dtype=np.float32)))
            if y_list[i] < min_temp:
                min_temp = y_list[i]
        loss = self.addn(loss_list)
        
        # loss = self.addn(y_list)
        return self.get_loss(loss)


if __name__ == "__main__":
    # 测试生成图
    P = 10
    batch_size = 1
    g = gene_random_instance(4)
    print(g)
    print(g.nodes)
    print(g.edges)
    # 测试hc hm and ansatz
    # hc = gene_Hc_circuit(g, 'g0')
    # print(hc)
    # hm = gene_Hm_circuit(g, 'm0')
    # print(hm)
    # ansatz = gene_qaoa_ansatz(g, 3)
    # print(ansatz)
    # 测试forward 过程
    qnn = gene_qaoa_layers(g, P)
    metaqaoa = MetaQAOA(T=5, QNN=qnn, lstm_layers=5)
    params = metaqaoa.trainable_params()
    for p in params:
        print(p.name)
    # theta = Tensor(3.14*np.random.random([batch_size, P, 2]).astype(np.float32))
    # h = Tensor(np.zeros([1*5, batch_size, 2]).astype(np.float32))
    # c = Tensor(np.zeros([1*5, batch_size, 2]).astype(np.float32))

    # theta, h, c, y_list = metaqaoa(theta, h, c)

    # for y in y_list:
    #     print(y)

    # loss_fn = MetaQAOALoss()
    # loss = loss_fn(y_list)
    # print(loss)