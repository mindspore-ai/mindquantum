# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:49:14 2022

赛题三：根据始末量子态，重构中间幺正操作

1. 采用卷积神经网络 
2. 对 mindquantum 源代码进行了修改，使其支持动态输入目标态

@author: Waikikilick
"""

import numpy as np
import mindspore as ms
from mindquantum import *
from mindspore import nn
from mindspore.nn import Adam, TrainOneStepCell
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)

train_x = np.load('./new_train_x.npy', allow_pickle=True)
train_y = np.load('./new_train_y.npy', allow_pickle=True)
eval_x = np.load('./new_eval_x.npy', allow_pickle=True)
eval_y = np.load('./new_eval_y.npy', allow_pickle=True)
test_x = np.load('./test_x.npy', allow_pickle=True)


# 自定义网络层
# 本来是不支持传入 target_state 的
class My_QLayer(nn.Cell):
    def __init__(self, expectation_with_grad, weight='normal'):
        super(My_QLayer, self).__init__()
        self.evolution = MQN2Ops(expectation_with_grad)
        weight_size = len(
            self.evolution.expectation_with_grad.ansatz_params_name)
        self.weight = Parameter(initializer(weight,
                                            weight_size,
                                            dtype=ms.float32),
                                name='ansatz_weight')

    def construct(self, x, target_state):
        return self.evolution(x, self.weight, target_state)


# encoder
def generate_encoder():
    enc_layer = Circuit()
    for i in range(3):
        enc_layer += U3(f'a{i}', f'b{i}', f'c{i}', i)
    coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])
    encoder = Circuit()
    for i in range(2):
        encoder += add_prefix(enc_layer, f'l{i}')
        encoder += coupling_layer
    return encoder, encoder.params_name


encoder, encoder_names = generate_encoder()
encoder.no_grad()


# ansatz
# 卷积神经网络，卷积核为 9b
def Ansatz(layer_num=10):
    ansatz = Circuit()
    for i in range(layer_num):
        ansatz += ansatz_layer(prefix=f'layer{i}')
    return ansatz


def ansatz_layer(prefix='0'):
    _circ = Circuit()
    _circ += BarrierGate()
    _circ += conv_circ(prefix='01', bit_up=0, bit_down=1)
    _circ += BarrierGate()
    _circ += conv_circ(prefix='12', bit_up=1, bit_down=2)
    _circ += BarrierGate()
    _circ += conv_circ(prefix='20', bit_up=2, bit_down=0)
    _circ += BarrierGate()
    _circ = add_prefix(_circ, prefix)
    return _circ


def conv_circ(prefix='0', bit_up=0, bit_down=1):
    _circ = Circuit()
    _circ += U3('theta00', 'phi00', 'lam00', bit_up)
    _circ += U3('theta01', 'phi01', 'lam01', bit_down)
    _circ += X.on(bit_down, bit_up)
    _circ += RY('theta10').on(bit_up)
    _circ += RZ('theta11').on(bit_down)
    _circ += X.on(bit_up, bit_down)
    _circ += RY('theta20').on(bit_up)
    _circ += X.on(bit_down, bit_up)
    _circ += U3('theta30', 'phi30', 'lam30', bit_up)
    _circ += U3('theta31', 'phi31', 'lam31', bit_down)
    _circ = add_prefix(_circ, prefix)
    return _circ


ansatz = Circuit()
ansatz = Ansatz(layer_num=1)
ansatz_names = ansatz.params_name
circuit = encoder.as_encoder() + ansatz
paras_name = circuit.params_name

ham = Hamiltonian(QubitOperator(''))
sim = Simulator('mqvector', 3)
sim_left = Simulator('mqvector', 3)

grad_ops = sim.get_expectation_with_grad(ham,
                                         circuit,
                                         circ_left=Circuit(),
                                         simulator_left=sim_left)
QuantumNet = My_QLayer(grad_ops)
opti = Adam(QuantumNet.trainable_params(),
            learning_rate=0.01)  # 需要优化的是Quantumnet中可训练的参数，学习率设为0.5
net = TrainOneStepCell(QuantumNet, opti)

for epoch in range(2):
    print('\nepoch is:', epoch)

    for i in range(len(train_x)):
        encoder_data = train_x[i].reshape((1, 18)).astype(np.float32)
        target_psi = train_y[i]
        res = net(ms.Tensor(encoder_data), ms.Tensor(target_psi))

        if i % 100 == 0:  # 每 100 步，用整个验证集验证一下
            print('step:', i, '训练保真度：', res)

# fid_list = []
# for x, y in zip(eval_x,eval_y):
#     paras = list(np.squeeze(x)) + list(QuantumNet.weight.asnumpy())
#     pr = dict(zip(paras_name, paras))
#     state = circuit.get_qs(pr=pr)
#     fid = np.abs(np.vdot(state, y))**2
#     fid_list.append(fid)
# print('验证集平均保真度：',np.mean(fid_list))

state_list = []
for x in eval_x:
    paras = list(np.squeeze(x)) + list(QuantumNet.weight.asnumpy())
    pr = dict(zip(paras_name, paras))
    state = circuit.get_qs(pr=pr)
    state_list.append(state)
np.save('./lala.npy', np.array(state_list))
print('导出完成啦！')

state_array = np.load('./lala.npy', allow_pickle=True)

acc = np.real(
    np.mean(
        [np.abs(np.vdot(bra, ket)) for bra, ket in zip(state_array, eval_y)]))
print('最终准确率：', np.mean(acc))

# state_list = []
# for x in test_x:
#     paras = list(np.squeeze(x)) + list(QuantumNet.weight.asnumpy())
#     pr = dict(zip(paras_name, paras))
#     state = circuit.get_qs(pr=pr)
#     state_list.append(state)
# np.save('./heihei.npy',np.array(state_list))
# print('导出完成啦！')

# 后面是在源代码中的修改内容

# # 1. 在 operations.py 中
# class MQN2Ops(nn.Cell):

#     def __init__(self, expectation_with_grad):
#         super(MQN2Ops, self).__init__()
#         _mode_check(self)
#         _check_grad_ops(expectation_with_grad)
#         self.expectation_with_grad = expectation_with_grad
#         self.shape_ops = P.Shape()

#     def extend_repr(self):
#         return self.expectation_with_grad.str

#     def construct(self, enc_data, ans_data, target_state): # target_state 是新加的
#         check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
#         check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
#         enc_data = enc_data.asnumpy()
#         ans_data = ans_data.asnumpy()
#         target_state = target_state.asnumpy() # 这行是新加的
#         f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data, target_state)
#         self.f = -f   # 新加了一个负号，由梯度下降变成了梯度上升
#         f = ms.Tensor(np.abs(f)**2, dtype=ms.float32)
#         self.g_enc = g_enc
#         self.g_ans = g_ans
#         return f

#     def bprop(self, enc_data, ans_data, target_state, out, dout): # 新加了 target_state
#         dout = dout.asnumpy()
#         enc_grad = 2 * np.real(np.einsum('smp,sm,sm->sp', self.g_enc, dout, np.conj(self.f)))
#         ans_grad = 2 * np.real(np.einsum('smp,sm,sm->p', self.g_ans, dout, np.conj(self.f)))
#         return ms.Tensor(enc_grad, dtype=ms.float32), ms.Tensor(ans_grad, dtype=ms.float32), ms.Tensor(np.zeros(target_state.shape), dtype=ms.float32)
#         # 返回值中新加了 ms.Tensor(np.zeros(target_state.shape), dtype=ms.float32)

# # 2. 在 simulator.py 中：

# def get_expectation_with_grad(self,
#                                   hams,
#                                   circ_right,
#                                   circ_left=None,
#                                   simulator_left=None,
#                                   encoder_params_name=None,
#                                   ansatz_params_name=None,
#                                   parallel_worker=None):

#         if isinstance(hams, Hamiltonian):
#             hams = [hams]
#         elif not isinstance(hams, list):
#             raise TypeError(f"hams requires a Hamiltonian or a list of Hamiltonian, but get {type(hams)}")
#         for h_tmp in hams:
#             _check_input_type("hams's element", Hamiltonian, h_tmp)
#             _check_hamiltonian_qubits_number(h_tmp, self.n_qubits)
#         _check_input_type("circ_right", Circuit, circ_right)
#         if circ_right.is_noise_circuit:
#             raise ValueError("noise circuit not support yet.")
#         non_hermitian = False
#         if circ_left is not None:
#             _check_input_type("circ_left", Circuit, circ_left)
#             if circ_left.is_noise_circuit:
#                 raise ValueError("noise circuit not support yet.")
#             non_hermitian = True
#         if simulator_left is not None:
#             _check_input_type("simulator_left", Simulator, simulator_left)
#             if self.backend != simulator_left.backend:
#                 raise ValueError(f"simulator_left should have the same backend as this simulator, \
# which is {self.backend}, but get {simulator_left.backend}")
#             if self.n_qubits != simulator_left.n_qubits:
#                 raise ValueError(f"simulator_left should have the same n_qubits as this simulator, \
# which is {self.n_qubits}, but get {simulator_left.n_qubits}")
#             non_hermitian = True
#         if non_hermitian and simulator_left is None:
#             simulator_left = self
#         if circ_left is None:
#             circ_left = circ_right
#         if circ_left.has_measure_gate or circ_right.has_measure_gate:
#             raise ValueError("circuit for variational algorithm cannot have measure gate")
#         if parallel_worker is not None:
#             _check_int_type("parallel_worker", parallel_worker)
#         if encoder_params_name is None and ansatz_params_name is None:
#             encoder_params_name = []
#             ansatz_params_name = [i for i in circ_right.params_name]
#             for i in circ_left.params_name:
#                 if i not in ansatz_params_name:
#                     ansatz_params_name.append(i)
#         if encoder_params_name is None:
#             encoder_params_name = []
#         if ansatz_params_name is None:
#             ansatz_params_name = []
#         _check_input_type("encoder_params_name", list, encoder_params_name)
#         _check_input_type("ansatz_params_name", list, ansatz_params_name)
#         for i in encoder_params_name:
#             _check_input_type("Element of encoder_params_name", str, i)
#         for i in ansatz_params_name:
#             _check_input_type("Element of ansatz_params_name", str, i)
#         s1 = set(circ_right.params_name) | set(circ_left.params_name)
#         s2 = set(encoder_params_name) | set(ansatz_params_name)
#         if s1 - s2 or s2 - s1:
#             raise ValueError("encoder_params_name and ansatz_params_name are different with circuit parameters")
#         circ_n_qubits = max(circ_left.n_qubits, circ_right.n_qubits)
#         if self.n_qubits < circ_n_qubits:
#             raise ValueError(f"Simulator has {self.n_qubits} qubits, but circuit has {circ_n_qubits} qubits.")
#         version = "both"
#         if not ansatz_params_name:
#             version = "encoder"
#         if not encoder_params_name:
#             version = "ansatz"

#         def grad_ops(*inputs):
#             if version == "both" and len(inputs) != 3: # 因需要多输入一个目标量子态，由原来的 !=2 变成了 != 3.
#                 raise ValueError("Need two inputs!")
#             if version in ("encoder", "ansatz") and len(inputs) != 1:
#                 raise ValueError("Need one input!")
#             if version == "both":
#                 _check_encoder(inputs[0], len(encoder_params_name))
#                 _check_ansatz(inputs[1], len(ansatz_params_name))
#                 batch_threads, mea_threads = _thread_balance(inputs[0].shape[0], len(hams), parallel_worker)
#                 inputs0 = inputs[0]
#                 inputs1 = inputs[1]
#                 target_state = inputs[2] # 多加了这一行
#             if version == "encoder":
#                 _check_encoder(inputs[0], len(encoder_params_name))
#                 batch_threads, mea_threads = _thread_balance(inputs[0].shape[0], len(hams), parallel_worker)
#                 inputs0 = inputs[0]
#                 inputs1 = np.array([])
#             if version == "ansatz":
#                 _check_ansatz(inputs[0], len(ansatz_params_name))
#                 batch_threads, mea_threads = _thread_balance(1, len(hams), parallel_worker)
#                 inputs0 = np.array([[]])
#                 inputs1 = inputs[0]
#             if non_hermitian:
#                 simulator_left.sim.set_qs(target_state) # 多加了这一行
#                 f_g1_g2 = self.sim.non_hermitian_measure_with_grad([i.get_cpp_obj() for i in hams],
#                                                                    [i.get_cpp_obj(hermitian=True) for i in hams],
#                                                                    circ_left.get_cpp_obj(),
#                                                                    circ_left.get_cpp_obj(hermitian=True),
#                                                                    circ_right.get_cpp_obj(),
#                                                                    circ_right.get_cpp_obj(hermitian=True), inputs0,
#                                                                    inputs1, encoder_params_name, ansatz_params_name,
#                                                                    batch_threads, mea_threads, simulator_left.sim)
#             else:
#                 f_g1_g2 = self.sim.hermitian_measure_with_grad([i.get_cpp_obj()
#                                                                 for i in hams], circ_right.get_cpp_obj(),
#                                                                circ_right.get_cpp_obj(hermitian=True), inputs0, inputs1,
#                                                                encoder_params_name, ansatz_params_name, batch_threads,
#                                                                mea_threads)
#             res = np.array(f_g1_g2)
#             if version == 'both':
#                 f = res[:, :, 0]
#                 g1 = res[:, :, 1:1 + len(encoder_params_name)]
#                 g2 = res[:, :, 1 + len(encoder_params_name):]
#                 return f, g1, g2
#             f = res[:, :, 0]
#             g = res[:, :, 1:]
#             return f, g

#         grad_wrapper = GradOpsWrapper(grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name,
#                                       parallel_worker)
#         s = f'{self.n_qubits} qubit' + ('' if self.n_qubits == 1 else 's')
#         s += f' {self.backend} VQA Operator'
#         grad_wrapper.set_str(s)
#         return grad_wrapper
