# -*- coding: utf-8 -*-
"""
Common functions.
"""
import numpy as np
from mindquantum.core import ParameterResolver, Circuit

def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict

def normal(state):
    return state / np.sqrt(np.abs(np.vdot(state, state)))
def calcu_acc(y, real_y):
    acc = np.real(
        np.mean([
            np.abs(np.vdot(normal(bra), ket))
            for bra, ket in zip(y, real_y)
        ]))
    return acc
def predict_acc(x, y, circ, w, epn, apn):
    app = param2dict(apn, w)
    o = []
    for i in range(len(y)):
        pp = dict(param2dict(epn, x[i]), **app)
        o.append(circ.get_qs(pr=pp))
    return calcu_acc(y, o)

def _pgate(gate, para, coeff, qobj):
    para = ParameterResolver({para:1})
    return gate(coeff * para).on(qobj)
def _phasegate(para, coeff, qobj):
    para = ParameterResolver({para:1})
    return Circuit().phase_shift(coeff * para, qobj)
