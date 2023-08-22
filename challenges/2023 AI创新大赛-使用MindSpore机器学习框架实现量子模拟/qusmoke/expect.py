"""Calculate expectation with specific hamiltonian."""

from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops

from .gates import X, Y, Z
from .gates import DTYPE


def get_hamiltonian_gate(ham: dict) -> tuple:
    """Get the hamiltonian from the dict.

    Args:
        ham (dict): The hamiltonian, which like {"Z0 X1": 1.0, "Z2": 2.0}, where
            keys are operations and values are corresponding coefficients.
    
    Return:
        (coef_list, op_list): The converted hamiltonian, which like ([1.0, 2.0], [[Z(0), X(1)], [Z(2)]])
    """
    coef_list = []
    op_list = []
    MEASURE_GATES = ['X', 'Y', 'Z', 'I']
    for op_str, coef in ham.items():
        op_str_list = op_str.split(' ')
        sub_op_list = []
        for s in op_str_list:
            s = s.strip()
            if len(s) >= 2:
                ss = s[0]
                obj = int(s[1:])
                assert ss in MEASURE_GATES, f"The measure gate should be in [{MEASURE_GATES}], but get {ss}."
                if ss == 'X':
                    sub_op_list.append(X(obj))
                elif ss == 'Y':
                    sub_op_list.append(Y(obj))
                elif ss == 'Z':
                    sub_op_list.append(Z(obj))
                else:
                    # for gate `I`, do nothing.
                    pass
        coef_list.append(float(coef))
        op_list.append(sub_op_list)
    return coef_list, op_list


class Expectation(nn.Cell):
    def __init__(self, ham):
        super().__init__()
        self.coef_list, self.op_list = get_hamiltonian_gate(ham)

    def construct(self, args):
        n_qubit, qs = args
        res = []
        for coef, ham in zip(self.coef_list, self.op_list):
            expect = self._item_expection(n_qubit, qs, ham)
            res.append(Tensor(coef, DTYPE) * expect)
        return ops.stack(res).reshape((1, -1))

    def _item_expection(self, n_qubit, qs, hams):
        qs2 = qs
        for ham in hams:
            _, qs2 = ham((n_qubit, qs2))
        return (qs[0] * qs2[0]).sum() + (qs[1] * qs2[1]).sum()
