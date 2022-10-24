import numpy as np
from mindquantum import H, RX, RY, RZ, XX, YY, ZZ 
from mindquantum import Circuit, add_prefix


def gene_tfim_encoder(n_qubits: int):
    """
    n_qubits: int can be 4, 8, 12
    return the encoder circuit for tfim question
    """
    half_num_qubits = n_qubits // 2 
    q1 = [t for t in range(n_qubits)][:: -1]
    q2 = [(t-1) % n_qubits for t in range(n_qubits)][:: -1]
    obj_qubits_list = [t for t in zip(q1, q2)]
    encoder = Circuit()
    encoder.un(H, n_qubits)
    for i in range(half_num_qubits):
        encoder.un(ZZ({f'theta_{i}': np.pi / 2}), obj_qubits_list)
        encoder.un(RX({f"theta_{i + half_num_qubits}": np.pi / 2}), n_qubits)
    return encoder


def gene_conv(prefix: str, obj_qubits: list):
    """
    prefix: conv block 的前缀，加在变量名上的
    obj_qubits: lists of obj_qubits, 顺序是无所谓的
    """
    conv = Circuit()
    conv += RX({'px0': np.pi / 2}).on(obj_qubits[0])
    conv += RX({'px1': np.pi / 2}).on(obj_qubits[1])
    conv += RY({'py0': np.pi / 2}).on(obj_qubits[0])
    conv += RY({'py1': np.pi / 2}).on(obj_qubits[1])
    conv += RZ({'pz0': np.pi / 2}).on(obj_qubits[0])
    conv += RZ({'pz1': np.pi / 2}).on(obj_qubits[1])
    conv += ZZ({'pzz': np.pi / 2}).on(obj_qubits)
    conv += YY({'pyy': np.pi / 2}).on(obj_qubits)
    conv += XX({'pxx': np.pi / 2}).on(obj_qubits)
    conv += RX({'px2': np.pi / 2}).on(obj_qubits[0])
    conv += RX({'px3': np.pi / 2}).on(obj_qubits[1])
    conv += RY({'py2': np.pi / 2}).on(obj_qubits[0])
    conv += RY({'py3': np.pi / 2}).on(obj_qubits[1])
    conv += RZ({'pz2': np.pi / 2}).on(obj_qubits[0])
    conv += RZ({'pz3': np.pi / 2}).on(obj_qubits[1])
    return add_prefix(conv, prefix=prefix)


def gene_pooling(prefix: str, obj_qubit: int, ctrl_qubit: int):
    """
    prefix: pooling block 加载参数前面的前缀
    obj_qubit: pooling 时候的目标比特
    ctrl_qubit: pooling 时候的控制比特
    """
    pooling = Circuit()
    pooling += RX({'px0': np.pi / 2}).on(obj_qubit)
    pooling += RX({'px1': np.pi / 2}).on(ctrl_qubit)
    pooling += RY({'py0': np.pi / 2}).on(obj_qubit)
    pooling += RY({'py1': np.pi / 2}).on(ctrl_qubit)
    pooling += RZ({'pz0': np.pi / 2}).on(obj_qubit)
    pooling += RZ({'pz1': np.pi / 2}).on(ctrl_qubit)
    # circuit for variational algorithm cannot have measure gate
    # pooling.measure(ctrl_qubit)
    pooling.x(obj_qubit, ctrl_qubit)
    pooling += RZ({'pz0': - np.pi / 2}).on(obj_qubit)
    pooling += RY({'py0': - np.pi / 2}).on(obj_qubit)
    pooling += RX({'px0': - np.pi / 2}).on(obj_qubit)

    return add_prefix(pooling, prefix=prefix)


def gene_ansatz(n_qubits: int):
    """
    根据num qubits生成对应的ansatz. n_qubits in [4, 8, 12]
    这一部分准备写通用的 QCNN 的网络结构。
    qubits_list = [i for i in range(n_qubits)]
    repetition = int(np.ceil(np.log2(n_qubits)))
    ansatz = Circuit()
    for layer in range(repetition):
        while 
    想法是qubits_list 里面两个一组往下排，pooling的qubits删除掉，一直循环下去
    如果遇到最后不是2的倍数，则把单个的空着，什么也不操作

    qubits数量少，直接人工搭建
    """
    ansatz = Circuit()

    if n_qubits == 4:
        ansatz += gene_conv("l1c1", [0, 1])
        ansatz += gene_conv("l1c2", [2, 3])
        ansatz += gene_pooling("l1p1", 0, 1)
        ansatz += gene_pooling("l1p2", 2, 3)
        ansatz += gene_conv("l2c1", [0, 2])
        ansatz += gene_pooling("l2p1", 0, 2)
    elif n_qubits == 8:
        ansatz += gene_conv("l1c1", [0, 1])
        ansatz += gene_conv("l1c2", [2, 3])
        ansatz += gene_conv("l1c3", [4, 5])
        ansatz += gene_conv("l1c4", [6, 7])
        ansatz += gene_pooling("l1p1", 0, 1)
        ansatz += gene_pooling("l1p2", 2, 3)
        ansatz += gene_pooling("l1p3", 4, 5)
        ansatz += gene_pooling("l1p4", 6, 7)
        ansatz += gene_conv("l2c1", [0, 2])
        ansatz += gene_conv("l2c2", [4, 6])
        ansatz += gene_pooling("l2p1", 0, 2)
        ansatz += gene_pooling("l2p2", 4, 6)
        ansatz += gene_conv("l3c1", [0, 4])
        ansatz += gene_pooling("l3p1", 0, 4)


    elif n_qubits == 12:
        ansatz += gene_conv("l1c1", [0, 1])
        ansatz += gene_conv("l1c2", [2, 3])
        ansatz += gene_conv("l1c3", [4, 5])
        ansatz += gene_conv("l1c4", [6, 7])
        ansatz += gene_conv("l1c5", [8, 9])
        ansatz += gene_conv("l1c6", [10, 11])
        ansatz += gene_pooling("l1p1", 0, 1)
        ansatz += gene_pooling("l1p2", 2, 3)
        ansatz += gene_pooling("l1p3", 4, 5)
        ansatz += gene_pooling("l1p4", 6, 7)
        ansatz += gene_pooling("l1p5", 8, 9)
        ansatz += gene_pooling("l1p6", 10, 11)
        ansatz += gene_conv("l2c1", [0, 2])
        ansatz += gene_conv("l2c2", [4, 6])
        ansatz += gene_conv("l2c3", [8, 10])
        ansatz += gene_pooling("l2p1", 0, 2)
        ansatz += gene_pooling("l2p2", 4, 6)
        ansatz += gene_pooling("l2p3", 8, 10)
        ansatz += gene_conv("l3c1", [0, 4])
        ansatz += gene_pooling("l3p1", 0, 4)
        ansatz += gene_conv("l4c1", [0, 6])
        ansatz += gene_pooling("l4p1", 0, 6)
    
    return ansatz




if __name__ == "__main__":
    n_qubit = 4
    encoder = gene_tfim_encoder(n_qubit)
    encoder.summary()
    print(encoder)
    # load 4 qubits data
    gamma, params = np.load(f"../data/{n_qubit}qbsdata.npy", allow_pickle=True)

    conv = gene_conv('l1c1', [0, 1])
    conv.summary()
    print(conv)
    pooling = gene_pooling("l2p1", 1, 0)
    pooling.summary()
    print(pooling)

    ansatz = gene_ansatz(n_qubits=4)
    ansatz.summary()