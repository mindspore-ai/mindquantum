"""
dqas_tool.py

This module provides functionalities related to DQAS (describe what DQAS stands for).
Include a brief description of what this module does and its main features.
"""
from typing import Union
import numpy as np
from mindquantum.core.gates import RY
from mindquantum.core.circuit import Circuit
import mindspore as ms
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator
from mindquantum.core.parameterresolver import PRGenerator
from mindspore import Tensor, ops
from mindquantum.core.operators import Hamiltonian
import torch
from mindquantum.framework import MQLayer, MQOps
import mindspore.numpy as mnp
from mindquantum.core.circuit import change_param_name, apply

# 损失函数定义 SoftmaxCrossEntropyWithLogits
loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


def sampling_from_structure(structures: np.array, num_layer: int, shape_parametized: int) -> np.array:
    '''
    本函数适用于 DQAS 的结构采样，根据结构概率生成采样结果
    structures: 结构概率
    num_layer: 结构层数
    shape_parametized: 参数化层数
    '''
    softmax = ops.Softmax()
    prob = softmax(ms.Tensor(structures, ms.float32))
    prob_np = prob.asnumpy()  # 将 MindSpore Tensor 转换为 NumPy 数组

    while True:
        samples = []
        for i in range(num_layer):
            sample = np.random.choice(prob_np[i].shape[0], p=prob_np[i])
            samples.append(sample)

        # 判断是否元素全都大于 shape_parametized
        if all(sample >= shape_parametized for sample in samples):
            continue  # 如果是，就重新采样
        else:
            break  # 如果不是，跳出循环

    return np.array(samples)


def dqas_accuracy_custom(ansatz: Circuit, network_params: np.array, x, y, n_qbits: int = 8):
    '''
    用来检验当前 Ansatz 的准确率
    ansatz: 量子电路
    network_params: 神经网络参数
    x: 输入数据
    y: 标签数据
    n_qbits: 量子比特数
    '''

    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    op = MQOps(grad_ops)
    raw_result = op(ms.Tensor(x), ms.Tensor(network_params).reshape(-1))
    softmax_pro = ops.Softmax()(raw_result)
    predicted_result = ops.Argmax()(softmax_pro)
    equal_elements = ops.equal(ms.Tensor(y), predicted_result)
    num_equal_elements = ops.reduce_sum(equal_elements.astype(ms.int32))
    acc = num_equal_elements.asnumpy() / x.shape[0]
    return acc


def nmf_gradient(structures: np.array, oh: ms.Tensor, num_layer: int, size_pool: int):
    """
    使用 MindSpore 实现蒙特卡洛梯度计算。
    """
    # Step 1: 获取选择的索引
    choice = ops.Argmax(axis=-1)(oh)
    # Step 2: 计算概率
    softmax = ops.Softmax(axis=-1)
    prob = softmax(ms.Tensor(structures))
    # Step 3: 获取概率矩阵中的值
    indices = mnp.stack((mnp.arange(num_layer, dtype=ms.int64), choice), axis=1)
    prob = ops.GatherNd()(prob, indices)
    # Step 4: 变换概率矩阵
    prob = prob.reshape(-1, 1)
    prob = ops.Tile()(prob, (1, size_pool))

    # Step 5: 生成蒙特卡洛梯度
    gradient = ops.TensorScatterAdd()(Tensor(-prob, ms.float64), indices, mnp.ones((num_layer,), dtype=ms.float64))
    return gradient


# 对向量化版本的封装
# nmf_gradient_vmap = ops.vmap(nmf_gradient, in_axes=(None, 0, None, None))


def best_from_structure(structures: np.array) -> Tensor:
    '''
    根据结构概率获取最优结构的索引
    '''
    return ops.Argmax(axis=-1)(ms.Tensor(structures))


def extract_parameterss(structure_parameters: np.array, candidate_index: np.array, shape_parametized: int):
    '''
    根据 候选index 从共享参数池中获取ansatz参数
    '''
    ansatz_parameters = []
    for layerindex, i in enumerate(candidate_index):
        if i >= shape_parametized:
            continue
        ansatz_parameters.append(structure_parameters[layerindex, i])

    return ansatz_parameters


def wash_pr(cir: Circuit, index: int):
    '''
    用来清理pr 的工具函数
    '''
    ansatz_before = cir
    if index is not None:
        name_map = dict(
            zip(
                ansatz_before.ansatz_params_name,
                [f'ansatz{index}-{i}' for i in range(len(ansatz_before.ansatz_params_name))],
            )
        )
    else:
        name_map = dict(
            zip(ansatz_before.ansatz_params_name, [f'ansatz{i}' for i in range(len(ansatz_before.ansatz_params_name))])
        )
    ansatz = change_param_name(ansatz_before, name_map)
    return ansatz


def vag_nnp_micro_minipool(
    structure_params: np.array,
    ansatz_params: np.array,
    paramerterized_pool: list[Circuit],
    unparamerterized_pool: list[Circuit],
    num_layer: int = 6,
    n_qbits: int = 8,
):
    """
    更新: 只在对应位置上更新nnp梯度
    用于计算梯度 ansatz_params关于 loss 的梯度
    value,grad_ansatz_params = vag(训练数据,标签数据)
    """
    if isinstance(structure_params, np.ndarray):
        mystp = ms.Tensor(structure_params, ms.float32)
    else:
        mystp = structure_params
    ansatz = mindspore_ansatz_micro_minipool(
        structure_p=mystp,
        parameterized_pool=paramerterized_pool,
        unparameterized_pool=unparamerterized_pool,
        num_layer=num_layer,
        n_qbits=n_qbits,
    )

    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    ansatz_pr = nnp_dealwith(structure_params=structure_params, network_params=ansatz_params)
    mylayer = MQLayer(grad_ops, ms.Tensor(ansatz_pr, ms.float64).reshape(-1))

    def forward_fn(encode_p, y_label):
        eval_obserables = mylayer(encode_p)
        loss = loss_fn(eval_obserables, y_label)
        return loss

    # nnp = ms.Tensor(ansatz_params).reshape(-1)
    grad_fn = ms.value_and_grad(fn=forward_fn, grad_position=None, weights=mylayer.trainable_params())

    return grad_fn


class WashingNamemap:
    '''
    为了更改ansatz 参数名称的迭代器
    '''

    def __init__(self, name: str = 'ansatz'):
        self.current = 0
        self.name = name

    def __next__(self):
        self.current += 1
        return self.name + str(self.current)


def mindspore_ansatz_micro_minipool(
    structure_p: np.array,
    parameterized_pool: list[Circuit],
    unparameterized_pool: list[Circuit],
    num_layer: int = 6,
    n_qbits: int = 8,
):
    """
    和 DQAS 文章描述的一致，生成权重线路
    更新了非参数化门的算符池引入
    structure_p:np.array DQAS中的权重参数,
    Ansatz_p:np.array  DQAS中的Ansatz参数,
    """
    if structure_p.shape[0] != num_layer:
        raise ValueError('structure_p shape must be equal to num_layer')

    if structure_p.shape[1] != len(parameterized_pool) + len(unparameterized_pool):
        raise ValueError('structure_p shape must be equal to size of pool')

    if isinstance(structure_p, np.ndarray):
        my_stp = ms.Tensor(structure_p, ms.float32)
    else:
        my_stp = structure_p

    prg = PRGenerator('encoder')
    nqbits = n_qbits
    encoder = Circuit()
    # encoder += UN(H, nqbits)
    for i in range(nqbits):
        encoder += RY(prg.new()).on(i)
    encoder = encoder.as_encoder()

    sub_ansatz = Circuit()
    my_washing_name_map = WashingNamemap()
    # print(my_stp.shape)
    for layer_index in range(my_stp.shape[0]):
        for op_index in range(my_stp.shape[1]):
            if my_stp[layer_index, op_index] == 0:
                continue
            if op_index < len(parameterized_pool):
                before_ansatz = parameterized_pool[op_index]
                # print(before_ansatz.ansatz_params_name)
                name_map = dict(
                    zip(
                        before_ansatz.ansatz_params_name,
                        [next(my_washing_name_map) for i in range(len(before_ansatz.ansatz_params_name))],
                    )
                )
                before_ansatz = change_param_name(circuit_fn=before_ansatz, name_map=name_map)
                sub_ansatz += before_ansatz
            else:
                sub_ansatz += unparameterized_pool[op_index - len(parameterized_pool)]

    whole_ansatz = Circuit()
    whole_ansatz += wash_pr(apply(sub_ansatz, [0, 1]), index=0)
    whole_ansatz += wash_pr(apply(sub_ansatz, [2, 3]), index=1)
    whole_ansatz += wash_pr(apply(sub_ansatz, [4, 5]), index=2)
    whole_ansatz += wash_pr(apply(sub_ansatz, [6, 7]), index=3)
    whole_ansatz += wash_pr(apply(sub_ansatz, [0, 2]), index=4)
    whole_ansatz += wash_pr(apply(sub_ansatz, [4, 6]), index=5)
    whole_ansatz += wash_pr(apply(sub_ansatz, [0, 4]), index=6)
    whole_ansatz = wash_pr(whole_ansatz, index=None)

    finnal_ansatz = encoder.as_encoder() + whole_ansatz.as_ansatz()
    return finnal_ansatz


def nnp_dealwith(structure_params: np.array, network_params: np.array, shape_parameterized: int = 2) -> np.array:
    ''' '
    从共享参数池里面获取ansatz参数
    '''
    candidate = best_from_structure(structure_params)
    ansatz_params = []
    for each_sub in range(7):
        for op_index, each_op in enumerate(candidate):
            if each_op >= shape_parameterized:
                continue
            # print(f'each_sub:{each_sub},op={each_op}')
            ansatz_params.append(network_params[each_sub, op_index, each_op, :])

    ansatz_params = np.array(ansatz_params).reshape(-1)
    return ansatz_params


def covert_ms2torch(mstensor: ms.Tensor) -> torch.Tensor:
    '''
    将mindspore的tensor转换为torch的tensor
    '''
    return torch.tensor(mstensor.asnumpy())


def zeroslike_grad_nnp_micro_minipool(
    batch_sturcture: Union[np.ndarray, ms.Tensor],
    grad_nnp: Union[np.ndarray, ms.Tensor],
    shape_parametized: int,
    ansatz_parameters: np.ndarray,
) -> np.ndarray:
    '''
    用于根据算出的梯度更新ansatz参数
    '''
    if isinstance(batch_sturcture, np.ndarray):
        mystp = ms.Tensor(batch_sturcture, ms.float32)
    else:
        mystp = batch_sturcture  # 如果 batch_sturcture 已经是 ms.Tensor 类型

    op_index = [ops.Argmax()(i) for i in mystp]
    # print(op_index)
    zeros_grad_nnp = np.zeros_like(ansatz_parameters)
    count = 0
    for each_sub in range(7):
        for index, i in enumerate(op_index):
            if i >= shape_parametized:
                continue
            zeros_grad_nnp[each_sub, index, i, :] = grad_nnp[count : count + 3]
            count += 3

    return zeros_grad_nnp
