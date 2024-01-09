# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""General IQP Encoding."""

import numpy as np

from mindquantum.algorithm.nisq._ansatz import Ansatz
from mindquantum.core.circuit import UN, Circuit, add_prefix, add_suffix
from mindquantum.core.gates import BARRIER, RZ, H, ParameterGate, X
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)


def _check_intrinsconeparagate(msg, gate_type):
    if not issubclass(gate_type, ParameterGate):
        raise TypeError(f"{msg} requires a type of {ParameterGate}, but get {gate_type}")


class IQPEncoding(Ansatz):
    r"""
    General IQP Encoding.

    For more information, please refer to `Supervised learning with quantum-enhanced feature
    spaces <https://www.nature.com/articles/s41586-019-0980-2>`_.

    Args:
        n_feature (int): The number of feature of data you want to encode with IQPEncoding.
        first_rotation_gate (ParameterGate): One of the rotation gate RX, RY or RZ.
        second_rotation_gate (ParameterGate): One of the rotation gate RX, RY or RZ.
        num_repeats (int): Number of encoding iterations.
        prefix (str): The prefix of parameters. Default: ``''``.
        suffix (str): The suffix of parameters. Default: ``''``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.nisq import IQPEncoding
        >>> iqp = IQPEncoding(3)
        >>> iqp.circuit
              ┏━━━┓ ┏━━━━━━━━━━━━┓
        q0: ──┨ H ┠─┨ RZ(alpha0) ┠───■─────────────────────────────■─────────────────────────────────────────
              ┗━━━┛ ┗━━━━━━━━━━━━┛   ┃                             ┃
              ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓
        q1: ──┨ H ┠─┨ RZ(alpha1) ┠─┨╺╋╸┠─┨ RZ(alpha0 * alpha1) ┠─┨╺╋╸┠───■─────────────────────────────■─────
              ┗━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛   ┃                             ┃
              ┏━━━┓ ┏━━━━━━━━━━━━┓                                     ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓
        q2: ──┨ H ┠─┨ RZ(alpha2) ┠─────────────────────────────────────┨╺╋╸┠─┨ RZ(alpha1 * alpha2) ┠─┨╺╋╸┠───
              ┗━━━┛ ┗━━━━━━━━━━━━┛                                     ┗━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛
        >>> iqp.circuit.params_name
        ['alpha0', 'alpha1', 'alpha2', 'alpha0 * alpha1', 'alpha1 * alpha2']
        >>> iqp.circuit.params_name
        >>> a = np.array([0, 1, 2])
        >>> iqp.data_preparation(a)
        array([0, 1, 2, 0, 2])
        >>> iqp.circuit.get_qs(pr=iqp.data_preparation(a))
        array([-0.28324704-0.21159186j, -0.28324704-0.21159186j,
                0.31027229+0.16950252j,  0.31027229+0.16950252j,
                0.02500938+0.35266773j,  0.02500938+0.35266773j,
                0.31027229+0.16950252j,  0.31027229+0.16950252j])
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        n_feature,
        first_rotation_gate=RZ,
        second_rotation_gate=RZ,
        num_repeats=1,
        prefix: str = '',
        suffix: str = '',
    ):
        """Initialize an IQPEncoding object."""
        _check_int_type("n_feature", n_feature)
        _check_value_should_not_less("n_feature", 1, n_feature)
        _check_int_type("num_repeats", num_repeats)
        _check_value_should_not_less("num_repeats", 1, num_repeats)
        _check_intrinsconeparagate("first_rotation_gate", first_rotation_gate)
        _check_intrinsconeparagate("second_rotation_gate", second_rotation_gate)
        _check_input_type('prefix', str, prefix)
        _check_input_type('suffix', str, suffix)

        self.n_feature = n_feature
        self.first_rotation_gate = first_rotation_gate
        self.second_rotation_gate = second_rotation_gate
        self.num_repeats = num_repeats
        self.prefix = prefix
        self.suffix = suffix
        super().__init__("IQPEncoding", n_feature)

    def _implement(self):  # pylint: disable=arguments-differ
        """Implement of iqp encoding ansatz."""
        self._circuit = UN(H, self.n_feature)
        repeat_unit = Circuit()
        for i in range(self.n_feature):
            repeat_unit += self.first_rotation_gate(f'alpha{i}').on(i)
        for i in range(1, self.n_feature):
            repeat_unit += X.on(i, i - 1)
            repeat_unit += self.second_rotation_gate(f'alpha{i - 1} * alpha{i}').on(i)
            repeat_unit += X.on(i, i - 1)
        repeat_unit += BARRIER
        self._circuit += repeat_unit * self.num_repeats
        if self.prefix:
            self._circuit = add_prefix(self._circuit, self.prefix)
        if self.suffix:
            self._circuit = add_suffix(self._circuit, self.suffix)

    def data_preparation(self, data):
        r"""
        Prepare the classical data into suitable dimension for IQPEncoding.

        The IQPEncoding ansatz provides an ansatz to encode classical data into quantum state.

        Suppose the origin data has :math:`n` features, then the output data will have :math:`2n-1` features,
        with first :math:`n` features will be the original data. For :math:`m > n`,

        .. math::

            \text{data}_m = \text{data}_{m - n} * \text{data}_{m - n - 1}

        Args:
            data ([list, numpy.ndarray]): The classical data need to encode with IQPEncoding ansatz.

        Returns:
            numpy.ndarray, the prepared data that is suitable for this ansatz.
        """
        _check_input_type("data", (list, np.ndarray), data)
        if isinstance(data, list):
            data = np.array(data)
        output = data * 1
        if len(output.shape) == 1:
            if output.shape[0] != self.n_feature:
                raise ValueError(
                    f"This iqp encoding requires {self.n_feature} features, but data has {output.shape[0]} features"
                )
            output = np.append(output, output[:-1] * output[1:])
            return output
        if len(output.shape) == 2:
            if output.shape[1] != self.n_feature:
                raise ValueError(
                    f"This iqp encoding requires {self.n_feature} features, but data has {output.shape[1]} features"
                )
            return np.append(output, output[:, :-1] * output[:, 1:], axis=1)
        raise ValueError(f"data need a one or two dimension array, but get dimension of {data.shape}")
