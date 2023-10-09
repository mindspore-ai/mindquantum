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
"""Type and value check helper."""

import numbers

import numpy as np

_num_type = (int, float, complex, np.int32, np.int64, np.float32, np.float64)


def _check_encoder(data, encoder_params_size):
    if not isinstance(data, np.ndarray):
        raise ValueError(f"encoder parameters need numpy array, but get {type(data)}")
    data_shape = data.shape
    if len(data_shape) != 2:
        raise ValueError("encoder data requires a two dimension numpy array")
    if data_shape[1] != encoder_params_size:
        raise ValueError(
            "encoder parameters size do not match with encoder parameters name, ",
            f"need {encoder_params_size} but get {data_shape[1]}.",
        )


def _check_ansatz(data, ansatz_params_size):
    """Check ansatz."""
    if not isinstance(data, np.ndarray):
        raise ValueError(f"ansatz parameters need numpy array, but get {type(data)}")
    data_shape = data.shape
    if len(data_shape) != 1:
        raise ValueError("ansatz data requires a one dimension numpy array")
    if data_shape[0] != ansatz_params_size:
        raise ValueError(
            "ansatz parameters size do not match with ansatz parameters name, "
            f"need {ansatz_params_size} but get {data_shape[0]}"
        )


def _check_hamiltonian_qubits_number(hamiltonian, sim_qubits):
    """Check hamiltonian qubits number."""
    # pylint: disable=import-outside-toplevel,cyclic-import
    from mindquantum.core.operators.hamiltonian import HowTo  # noqa: E402

    if hamiltonian.how_to != HowTo.ORIGIN:
        if hamiltonian.n_qubits != sim_qubits:
            raise ValueError(
                f"Hamiltonian qubits is {hamiltonian.n_qubits}, not match with simulator qubits number {sim_qubits}"
            )
    else:
        if hamiltonian.n_qubits > sim_qubits:
            raise ValueError(f"Hamiltonian qubits is {hamiltonian.n_qubits}, which is bigger than simulator qubits.")


def _check_np_dtype(dtype):
    """Check dtype is a valid numpy dtype."""
    np.array([0], dtype=dtype)


def _check_seed(seed):
    """Check seed."""
    _check_int_type("seed", seed)
    _check_value_should_between_close_set("seed", 0, 2**23, seed)


def _check_input_type(arg_msg, require_type, arg):
    """Check input type."""
    if not isinstance(arg, require_type):
        raise TypeError(f"{arg_msg} requires a {require_type}, but get {type(arg)}")


def _check_int_type(args_msg, arg):
    """Check int type."""
    if not isinstance(arg, (int, np.int64)) or isinstance(arg, bool):
        raise TypeError(f"{args_msg} requires an int, but get {type(arg)}")


def _check_value_should_not_less(arg_msg, require_value, arg):
    """Check value should not less."""
    if arg < require_value:
        raise ValueError(f'{arg_msg} should be not less than {require_value}, but get {arg}')


def _check_value_should_between_close_set(arg_ms, min_value, max_value, arg):
    """Check value should between."""
    if arg < min_value or arg > max_value:
        raise ValueError(f"{arg_ms} should between {min_value} and {max_value}, but get {arg}")


def _check_and_generate_pr_type(pr, names=None):
    """Check and generate PR type."""
    # pylint: disable=import-outside-toplevel,cyclic-import
    from ..core.parameterresolver import ParameterResolver

    if isinstance(pr, _num_type):
        if len(names) != 1:
            raise ValueError(f"number of given parameters value is less than parameters ({len(names)})")
        pr = np.array([pr])
    _check_input_type('parameter', (ParameterResolver, np.ndarray, list, dict), pr)
    if isinstance(pr, dict):
        pr = ParameterResolver(pr)
    elif isinstance(pr, (np.ndarray, list)):
        pr = np.array(pr)
        if len(pr) != len(names) or len(pr.shape) != 1:
            raise ValueError(f"given parameter value size ({pr.shape}) not match with parameter size ({len(names)})")
        pr = ParameterResolver(dict(zip(names, pr)))
    if isinstance(pr, ParameterResolver):
        if names is not None:
            for n in names:
                if n not in pr:
                    raise ValueError(f"Parameter {n} not in given parameter resolver.")
    return pr


def _check_number_type(arg_msg, arg):
    """Check number type."""
    if not isinstance(arg, numbers.Number):
        raise TypeError(f"{arg_msg} requires a number, but get {type(arg)}")


def _check_gate_type(gate):
    # pylint: disable=import-outside-toplevel,cyclic-import
    from ..core.gates.basic import BasicGate

    if not isinstance(gate, BasicGate):
        raise TypeError(f"Require a quantum gate, but get {type(gate)}")


def _check_gate_has_obj(gate):
    from ..core.gates.basicgate import (  # pylint: disable=import-outside-toplevel,cyclic-import
        BarrierGate,
    )

    if not isinstance(gate, BarrierGate):
        if not gate.obj_qubits:
            raise ValueError("Gate should act on some qubits first.")


def _check_qubit_id(qubit_id):
    if not isinstance(qubit_id, (int, np.int64)):
        raise TypeError(f"Qubit should be a non negative int, but get {type(qubit_id)}!")
    if qubit_id < 0:
        raise ValueError(f"Qubit should be non negative int, but get {qubit_id}!")


def _check_obj_and_ctrl_qubits(obj_qubits, ctrl_qubits):
    if set(obj_qubits) & set(ctrl_qubits):
        raise ValueError("obj_qubits and ctrl_qubits cannot have same qubits.")
    if len(set(obj_qubits)) != len(obj_qubits):
        raise ValueError("obj_qubits cannot have same qubits")
    if len(set(ctrl_qubits)) != len(ctrl_qubits):
        raise ValueError("ctrl_qubits cannot have same qubits")


def _check_control_num(ctrl_qubits, require_n):
    from .quantifiers import (  # pylint: disable=import-outside-toplevel,cyclic-import
        s_quantifier,
    )

    if len(ctrl_qubits) != require_n:
        raise RuntimeError(f"requires {s_quantifier(require_n,'control qubit')}, but get {len(ctrl_qubits)}")


def _require_package(pkg_name: str, low_version: str = None, high_version: str = None):
    if low_version is not None and high_version is not None:
        raise ImportError(
            f"Requires {pkg_name}, please install with 'pip install \"{pkg_name}>={low_version},<={high_version}\"'"
        )
    if low_version is not None:
        raise ImportError(f"Requires {pkg_name}, please install with 'pip install \"{pkg_name}>={low_version}\"'")
    if high_version is not None:
        raise ImportError(f"Requires {pkg_name}, please install with 'pip install \"{pkg_name}<={high_version}\"'")
    raise ImportError(f"Requires {pkg_name}, please install with 'pip install {pkg_name}'")


def _check_mq_type(dtype):
    # pylint: disable=import-outside-toplevel,cyclic-import
    from mindquantum.dtype.dtype import mq_number_type

    if dtype not in mq_number_type:
        raise ValueError(f"dtype only support {mq_number_type}, but get {dtype}")
