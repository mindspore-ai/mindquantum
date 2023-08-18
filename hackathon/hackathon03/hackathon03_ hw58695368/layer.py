# -*- coding: utf-8 -*-
"""
Fine tune the function of `mindquantum.framework.operations.MQOps`.

@NE
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindquantum.framework.operations import MQOps, check_enc_input_shape, check_ans_input_shape
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

class MQOps_re(MQOps):
    """
    """
    def __init__(self, expectation_with_grad):
        super(MQOps_re, self).__init__(expectation_with_grad)

    def construct(self, enc_data, ans_data):
        check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        f = ms.Tensor(1-np.real(np.abs(f)), dtype=ms.float32)
        self.g_enc = np.real(g_enc)
        self.g_ans = np.real(g_ans)
        return f
class MQLayer_re(nn.Cell):
    """
    used in `solution_td1.py`
    <Temporary Discard>
    """
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQLayer_re, self).__init__()
        self.evolution = MQOps_re(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get {weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, x):
        return self.evolution(x, self.weight)
'''
class MQOps_re2(MQOps):
    """
    """
    def __init__(self, expectation_with_grad):
        super(MQOps_re2, self).__init__(expectation_with_grad)

    def construct(self, enc_data, ans_data):
        check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        f = ms.Tensor(np.abs(f), dtype=ms.complex128)
        self.g_enc = np.real(g_enc)
        self.g_ans = np.real(g_ans)
        return f
class MQLayer_re2(nn.Cell):
    """
    <Unused>
    <Temporary Discard>
    """
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQLayer_re2, self).__init__()
        self.evolution = MQOps_re2(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get {weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, x):
        return self.evolution(x, self.weight)
'''
class MQOps_re_sp(MQOps):
    """
    """
    def __init__(self, expectation_with_grad, sim_m):
        super(MQOps_re_sp, self).__init__(expectation_with_grad)
        self.sim_m = sim_m

    def construct(self, enc_data, ans_data, ver_data):
        #check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
        #check_enc_input_shape(ver_data, self.shape_ops(ver_data), len(self.expectation_with_grad.encoder_params_name))
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        ver_data = ver_data.asnumpy()
        f, self.g_enc, self.g_ans = [], [], []
        dn = enc_data.shape[0]
        for i in range(dn):
            self.sim_m.set_qs(ver_data[i])
            e_d = np.array([enc_data[i]])
            f_, g_enc_, g_ans_ = self.expectation_with_grad(e_d, ans_data)
            f.append(np.abs(f_[0]))
            self.g_enc.append(g_enc_[0])
            self.g_ans.append(g_ans_[0])
        f = ms.Tensor(1-np.real(f), dtype=ms.float32)
        self.g_enc = np.real(self.g_enc)
        self.g_ans = np.real(self.g_ans)
        return f
    def bprop(self, enc_data, ans_data, ver_data, out, dout):
        #print(enc_data, ans_data, ver_data, out, dout)
        dout = dout.asnumpy()
        enc_grad = np.einsum('smp,sm->sp', self.g_enc, dout)
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return ms.Tensor(enc_grad, dtype=ms.float32), ms.Tensor(ans_grad, dtype=ms.float32)
    def grad_exec(self, enc_data, ans_data, ver_data):
        f = self.construct(enc_data, ans_data, ver_data)
        dout = [[1.] for i in range(enc_data.shape[0])]
        eg, ag = self.bprop(enc_data, ans_data, ver_data, [], ms.Tensor(dout))
        return f, eg, ag
class MQLayer_re_sp(nn.Cell):
    """
    used in `solution_td2.py`
    """
    def __init__(self, expectation_with_grad, sim_m, weight='normal'):
        super(MQLayer_re_sp, self).__init__()
        self.evolution = MQOps_re_sp(expectation_with_grad, sim_m)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get {weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, x, y):
        return self.evolution(x, self.weight, y)
