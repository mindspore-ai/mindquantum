import mindspore as ms
import mindspore.numpy as np
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindspore.nn import LSTM

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

iterations = 10
class RnnModel(ms.nn.Cell):
    def __init__(self, num_layers=4, depth=4):
        super().__init__()
        self.n_layers = num_layers
        self.cell = LSTM(depth * 2 + 1, hidden_size=depth * 2, num_layers=2 * self.n_layers)
        self.depth = depth

    def construct(self, inputs, graph_cost):
        prev_cost = inputs[0]
        prev_params = inputs[1]
        prev_h = inputs[2]
        prev_c = inputs[3]

        # Concatenate the previous parameters and previous cost to create new input
        new_input = np.concatenate([prev_cost, prev_params], axis=-1)

        # Call the LSTM cell, which outputs new values for the parameters along
        # with new internal states h and c

        new_params, [new_h, new_c] = self.cell(new_input, hx=(prev_h, prev_c))
        # Reshape the parameters to correctly match those expected by PennyLane
        _params = np.reshape(new_params, new_shape=(2*self.depth,))
        # Evaluate the cost using new angles
        _cost = graph_cost(_params)

        # Reshape to be consistent with other tensors
        new_cost = np.reshape(np.array(_cost, dtype=np.float32), new_shape=(1, 1, 1))

        return [new_cost, new_params, new_h, new_c]


class Model(ms.nn.Cell):
    def __init__(self, num_layers=4, depth=4):
        super().__init__()
        self.rnn_iteration = RnnModel(num_layers, depth)
        self.n_layers = num_layers
        self.graph_cost = None
        self.depth = depth

    def construct(self, inputs):
        """Creates the recurrent loop for the Recurrent Neural Network."""
        graph_cost = self.graph_cost

        initial_cost, initial_params, initial_h, initial_c = get_initial_value(inputs, self.n_layers, self.depth)

        out = [[initial_cost, initial_params, initial_h, initial_c]]
        for _ in range(iterations):
            out.append(self.rnn_iteration(out[-1], graph_cost))

        loss = np.mean(
            np.concatenate(
                [0.1 * i * out[i][0] for i in range(1, len(out))]
            )
        )
        return loss

    def predict(self, inputs, graph_cost):
        """Creates the recurrent loop for the Recurrent Neural Network."""

        # Initialize starting all inputs (cost, parameters, hidden states) as zeros.
        initial_cost, initial_params, initial_h, initial_c = get_initial_value(inputs, self.n_layers, self.depth)

        # We perform five consecutive calls to 'rnn_iteration', thus creating the
        # recurrent loop. More iterations lead to better results, at the cost of
        # more computationally intensive simulations.
        out = [[initial_cost, initial_params, initial_h, initial_c]]
        for _ in range(iterations):
            out.append(self.rnn_iteration(out[-1], graph_cost))

        # This cost function takes into account the cost from all iterations,
        # but using different weights.
        return [np.reshape(out[i][1], new_shape=(2 * self.depth,)) for i in range(1, len(out))]


def get_initial_value(inputs, n_layers, depth):
    initial_cost, initial_params = inputs
    initial_cost = np.reshape(ms.Tensor(initial_cost, dtype=ms.float32), new_shape=(1, 1, 1))
    initial_params = np.reshape(ms.Tensor(initial_params, dtype=ms.float32), new_shape=(1, 1, 2 * depth))
    initial_h = np.zeros(shape=(2 * n_layers, 1, 2 * depth))
    initial_c = np.zeros(shape=(2 * n_layers, 1, 2 * depth))
    return initial_cost, initial_params, initial_h, initial_c


class TrainOneStepCell(ms.nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        """参数初始化"""
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        # 使用tuple包装weight
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        # 定义梯度函数
        self.grad = ms.ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, data, graph_cost):
        """构建训练过程"""
        weights = self.weights
        self.network.graph_cost = graph_cost
        loss = self.network(data)
        # 为反向传播设定系数
        sens = ms.ops.Fill()(ms.ops.DType()(loss), ms.ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, sens)
        return loss, self.optimizer(grads)


def main(Jc_dict, p, Nq=14):
    """
        The main function you need to change!!!
    Args:
        Jc_dict (dict): the ising model
        p (int): the depth of qaoa circuit
    Returns:
        gammas (Union[numpy.ndarray, List[float]]): the gamma parameters, the length should be equal to depth p.
        betas (Union[numpy.ndarray, List[float]]): the beta parameters, the length should be equal to depth p.
    """
    k = max([len(key) for key in Jc_dict.keys()])
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k, 6)
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3 + 2 * p:
                    gammas = np.array([float(new_row[i]) for i in range(3, 3 + p)])
                    betas = np.array([float(new_row[i]) for i in range(3 + p, 3 + 2 * p)])
    # res = get_params(Jc_dict, Nq, depth=p)
    # return res[[0, 2, 4, 6]], res[[1, 3, 5, 7]]

    fn = get_cost_fn(Jc_dict, Nq, depth=p)
    params = np.concatenate([np.array([gammas[i], betas[i]]) for i in range(p)], axis=-1)
    initial_cost = fn(params)
    net = Model(4*p, p)
    # opt = ms.nn.SGD(net.trainable_params(), learning_rate=1e-4)
    ms.load_checkpoint(f'model3/{p}_model.ckpt', net)
    # train_net = TrainOneStepCell(net, opt, )

    net.set_train(False)
    res = net.predict([initial_cost, params], fn)[-1].asnumpy()
    idx1, idx2 = [2*i for i in range(p)], [2*i+1 for i in range(p)]
    return res[idx1], res[idx2]


def get_cost_fn(Jc_dict, Nq, depth):
    from mindquantum.core.circuit import Circuit, UN
    from mindquantum.core.gates import H, RX
    from mindquantum.core.operators import TimeEvolution, QubitOperator
    from mindquantum.core.parameterresolver import PRGenerator
    from mindquantum import MQAnsatzOnlyOps

    def build_hb(n, para=None):
        hb = Circuit()
        for i in range(n):
            if type(para) is str:
                hb += RX(dict([(para, 2)])).on(i)  # 对每个节点作用RX门
            else:
                hb += RX(para * 2).on(i)
        return hb

    def build_hc_high(ham, para):
        hc = Circuit()  # 创建量子线路
        hc += TimeEvolution(ham, time=para * (-1)).circuit
        return hc

    def build_ham_high(Jc_dict):
        ham = QubitOperator()
        for key, value in Jc_dict.items():
            nq = len(key)
            ops = QubitOperator(f'Z{key[0]}')
            for i in range(nq - 1):
                ops *= QubitOperator(f'Z{key[i + 1]}')  # 生成哈密顿量Hc
            ham += ops * value
        return ham

    def qaoa_hubo(nq, p=1):
        circ = Circuit()
        betas = PRGenerator('betas')
        gammas = PRGenerator('gammas')
        circ += UN(H, range(nq))
        for i in range(p):
            circ += build_hc_high(hamop, gammas.new())
            circ += build_hb(nq, para=betas.new())
        return circ

    hamop = build_ham_high(Jc_dict)

    class MQLayer(ms.nn.Cell):
        def __init__(self, expectation_with_grad):
            super(MQLayer, self).__init__()
            self.evolution = MQAnsatzOnlyOps(expectation_with_grad)

        def construct(self, params):
            return self.evolution(params)

    return MQLayer(
        Simulator('mqvector', n_qubits=Nq).get_expectation_with_grad(
            Hamiltonian(hamop), qaoa_hubo(Nq, p=depth)
        )
    )


def prepare_data():
    import json
    import csv
    def load_data(filename):
        '''
        Load the data for scoring.
        Args:
            filename (str): the name of ising model.
        Returns:
            Jc_dict (dict): new form of ising model for simplicity like {(0,): 1, (1, 2, 3): -1.}
        '''
        data = json.load(open(filename, 'r'))
        Jc_dict = {}
        for item in range(len(data['c'])):
            Jc_dict[tuple(data['J'][item])] = data['c'][item]
        return Jc_dict

    Nq = 12
    total_data = []
    for depth in [
        # 4,
        8]:
        data_list = []
        p = depth
        for propotion in [0.3, 0.9]:
            for k in range(2, 5):
                for coef in ['std', 'uni', 'bimodal']:
                    for r in range(5):
                        Jc_dict = load_data(f"data/k{k}/{coef}_p{propotion}_{r}.json")
                        fn = get_cost_fn(Jc_dict, Nq, depth=depth)
                        k = max([len(key) for key in Jc_dict.keys()])

                        # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
                        k = min(k, 6)
                        with open('utils/transfer_data.csv', 'r') as csv_file:
                            reader = csv.reader(csv_file)
                            for row in reader:
                                if (row[0]) == str(k):
                                    new_row = [item for item in row if item != '']
                                    length = len(new_row)
                                    if length == 3 + 2 * p:
                                        gammas = np.array([float(new_row[i]) for i in range(3, 3 + p)])
                                        betas = np.array([float(new_row[i]) for i in range(3 + p, 3 + 2 * p)])
                        params = np.concatenate([np.array([gammas[i], betas[i]]) for i in range(depth)], axis=-1)
                        initial_cost = fn(params)
                        data_list.append(([initial_cost, params], fn))
        total_data.append(data_list)
    return total_data


def train(data_list):
    import random
    from sklearn.model_selection import train_test_split

    for i, depth in enumerate([
        # 4,
        8]):
        net = Model(4*depth, depth)
        opt = ms.nn.Adam(net.trainable_params(), learning_rate=2e-4)
        ms.load_checkpoint(f'model3/{depth}_model.ckpt', net)
        train_net = TrainOneStepCell(net, opt, )
        train_list, test_list = train_test_split(data_list[i], test_size=0.3)
        epochs = 100
        acc = 0
        print(depth, len(train_list))
        for e in range(epochs):
            random.shuffle(train_list)
            for i, (train_data, fn) in enumerate(train_list):
                loss, _ = train_net(train_data, fn)
                if i % 30 == 0:
                    print(e, i, loss)
            s = 0
            train_net.set_train(False)
            for test_data, fn in test_list:
                res = net.predict(test_data, fn)[-1]  # .asnumpy()
                E = fn(res)
                s += -E
                # print(E)
            if s > acc:
                ms.save_checkpoint(train_net, f'model3/{depth}_model.ckpt')
            acc = max(s, acc)
            print(acc)
            train_net.set_train(True)
        # break
    return


if __name__ == "__main__":
    train(prepare_data())
    ms.nn.TransformerEncoder
    pass
