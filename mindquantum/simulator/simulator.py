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
"""Simulator."""
import warnings

import numpy as np

import mindquantum.mqbackend as mb

from ..core.circuit import Circuit
from ..core.gates import BarrierGate, BasicGate, Measure, MeasureResult
from ..core.operators import Hamiltonian
from ..core.operators.hamiltonian import HowTo
from ..core.parameterresolver import ParameterResolver
from ..utils.string_utils import ket_string
from ..utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_input_type,
    _check_int_type,
    _check_seed,
    _check_value_should_not_less,
)

SUPPORTED_SIMULATOR = ['projectq']


def get_supported_simulator():
    """
    Get simulator name that supported by MindQuantum.

    Returns:
        list, The supported simulator list.
    """
    return SUPPORTED_SIMULATOR


class Simulator:
    """
    Quantum simulator that simulate quantum circuit.

    Args:
        backend (str): which backend you want. The supported backend can be found
            in SUPPORTED_SIMULATOR
        n_qubits (int): number of quantum simulator.
        seed (int): the random seed for this simulator, if None, seed will generate
            by `numpy.random.randint`. Default: None.

    Raises:
        TypeError: if `backend` is not str.
        TypeError: if `n_qubits` is not int.
        TypeError: if `seed` is not int.
        ValueError: if `backend` is not supported.
        ValueError: if `n_qubits` is negative.
        ValueError: if `seed` is less than 0 or great than 2**23 - 1.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('projectq', 2)
        >>> sim.apply_circuit(qft(range(2)))
        >>> sim.get_qs()
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
    """

    def __init__(self, backend, n_qubits, seed=None):
        """Initialize a Simulator object."""
        _check_input_type('backend', str, backend)
        _check_int_type('n_qubits', n_qubits)
        _check_value_should_not_less('n_qubits', 0, n_qubits)
        if seed is None:
            seed = np.random.randint(1, 2**23)
        _check_seed(seed)
        if backend not in SUPPORTED_SIMULATOR:
            raise ValueError(f"backend {backend} not supported, now we support {SUPPORTED_SIMULATOR}!")
        self.backend = backend
        self.seed = seed
        self.n_qubits = n_qubits
        if backend == 'projectq':
            self.sim = mb.projectq(seed, n_qubits)

    def copy(self):
        """
        Copy this simulator.

        Returns:
            Simulator, a copy version of this simulator.

        Examples:
            >>> from mindquantum.core.gates import RX
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_gate(RX(1).on(0))
            >>> sim.flush()
            >>> sim2 = sim.copy()
            >>> sim2.apply_gate(RX(-1).on(0))
            >>> sim2
            projectq simulator with 1 qubit (little endian).
            Current quantum state:
            1¦0⟩
        """
        sim = Simulator(self.backend, self.n_qubits, self.seed)
        sim.sim = self.sim.copy()
        return sim

    def __str__(self):
        """Return a string representation of the object."""
        state = self.get_qs()
        ret = f"{self.backend} simulator with {self.n_qubits} qubit{'s' if self.n_qubits > 1 else ''} (little endian)."
        ret += "\nCurrent quantum state:\n"
        if self.n_qubits < 4:
            ret += '\n'.join(ket_string(state))
        else:
            ret += state.__str__()
        return ret

    def __repr__(self):
        """Return a string representation of the object."""
        return self.__str__()

    def reset(self):
        """
        Reset simulator to zero state.

        Examples:
            >>> from mindquantum.algorithm.library import qft
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.reset()
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
        """
        self.sim.reset()

    def flush(self):
        """
        Flush gate that works for projectq simulator.

        The projectq simulator will cache several gate and fushion these gate into a bigger gate, and than act on the
        quantum state. The flush command will ask the simulator to fushion currently stored gate and act on the quantum
        state.

        Examples:
            >>> from mindquantum.core.gates import H
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_gate(H.on(0))
            >>> sim.flush()
        """
        if self.backend == 'projectq':
            self.sim.run()

    def apply_gate(self, gate, pr=None, diff=False):
        """
        Apply a gate on this simulator, can be a quantum gate or a measurement operator.

        Args:
            gate (BasicGate): The gate you want to apply.
            pr (Union[numbers.Number, numpy.ndarray, ParameterResolver, list]): The
                parameter for parameterized gate. Default: None.
            diff (bool): Whether to apply the derivative gate on this simulator. Default: False.

        Returns:
            int or None, if the gate if a measure gate, then return a collapsed state, Otherwise
            return None.

        Raises:
            TypeError: if `gate` is not a BasicGate.
            ValueError: if any qubit of `gate` is higher than simulator qubits.
            ValueError: if `gate` is parameterized, but no parameter supplied.
            TypeError: the `pr` is not a ParameterResolver if `gate` is parameterized.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.gates import RY, Measure
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_gate(RY('a').on(0), np.pi/2)
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
            >>> sim.apply_gate(Measure().on(0))
            1
            >>> sim.get_qs()
            array([0.+0.j, 1.+0.j])
        """
        _check_input_type('gate', BasicGate, gate)
        if not isinstance(gate, BarrierGate):
            gate_max = max(max(gate.obj_qubits, gate.ctrl_qubits))
            if self.n_qubits < gate_max:
                raise ValueError(f"qubits of gate {gate} is higher than simulator qubits.")
            if isinstance(gate, Measure):
                return self.sim.apply_measure(gate.get_cpp_obj())
            if pr is None:
                if gate.parameterized:
                    raise ValueError("apply a parameterized gate needs a parameter_resolver")
                self.sim.apply_gate(gate.get_cpp_obj())
            else:
                pr = _check_and_generate_pr_type(pr, gate.coeff.params_name)
                self.sim.apply_gate(gate.get_cpp_obj(), pr.get_cpp_obj(), diff)
        return None

    def apply_circuit(self, circuit, pr=None):
        """
        Apply a circuit on this simulator.

        Args:
            circuit (Circuit): The quantum circuit you want to apply on this simulator.
            pr (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): The
                parameter resolver for this circuit. If the circuit is not parameterized,
                this arg should be None. Default: None.

        Returns:
            MeasureResult or None, if the circuit has measure gate, then return a MeasureResult,
            otherwise return None.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.gates import H
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(Circuit().un(H, 2))
            >>> sim.apply_circuit(Circuit().ry('a', 0).ry('b', 1), np.array([1.1, 2.2]))
            >>> sim
            projectq simulator with 2 qubits  (little endian).
            Current quantum state:
            -0.0721702531972066¦00⟩
            -0.30090405886869676¦01⟩
            0.22178317006196263¦10⟩
            0.9246947752567126¦11⟩
            >>> sim.apply_circuit(Circuit().measure(0).measure(1))
            shots: 1
            Keys: q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
            ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                     11│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                       │
            {'11': 1}
        """
        _check_input_type('circuit', Circuit, circuit)
        if self.n_qubits < circuit.n_qubits:
            raise ValueError(f"Circuit has {circuit.n_qubits} qubits, which is more than simulator qubits.")
        if circuit.has_measure_gate:
            res = MeasureResult()
            res.add_measure(circuit.all_measures.keys())
        if circuit.params_name:
            if pr is None:
                raise ValueError("Applying a parameterized circuit needs a parameter_resolver")
            pr = _check_and_generate_pr_type(pr, circuit.params_name)
        else:
            pr = ParameterResolver()
        if circuit.has_measure_gate:
            samples = np.array(
                self.sim.apply_circuit_with_measure(circuit.get_cpp_obj(), pr.get_cpp_obj(), res.keys_map)
            )
            samples = samples.reshape((1, -1))
            res.collect_data(samples)
            return res
        if circuit.params_name:
            self.sim.apply_circuit(circuit.get_cpp_obj(), pr.get_cpp_obj())
        else:
            self.sim.apply_circuit(circuit.get_cpp_obj())
        return None

    def sampling(self, circuit, pr=None, shots=1, seed=None):
        """
        Samping the measure qubit in circuit. Sampling do not change the origin quantum state of this simulator.

        Args:
            circuit (Circuit): The circuit that you want to evolution and do sampling.
            pr (Union[None, dict, ParameterResolver]): The parameter
                resolver for this circuit, if this circuit is a parameterized circuit.
                Default: None.
            shots (int): How many shots you want to sampling this circuit. Default: 1
            seed (int): Random seed for random sampling. If None, seed will be a random
                int number. Default: None.

        Returns:
            MeasureResult, the measure result of sampling.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.gates import Measure
            >>> from mindquantum.simulator import Simulator
            >>> circ = Circuit().ry('a', 0).ry('b', 1)
            >>> circ += Measure('q0_0').on(0)
            >>> circ += Measure('q0_1').on(0)
            >>> circ += Measure('q1').on(1)
            >>> sim = Simulator('projectq', circ.n_qubits)
            >>> res = sim.sampling(circ, {'a': 1.1, 'b': 2.2}, shots=100, seed=42)
            >>> res
            shots: 100
            Keys: q1 q0_1 q0_0│0.00   0.122       0.245       0.367        0.49       0.612
            ──────────────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                           000│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                              │
                           011│▒▒▒▒▒▒▒▒▒
                              │
                           100│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                              │
                           111│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                              │
            {'000': 18, '011': 9, '100': 49, '111': 24}
        """
        if not circuit.all_measures.map:
            raise ValueError("circuit must have at least one measurement gate.")
        _check_input_type("circuit", Circuit, circuit)
        if self.n_qubits < circuit.n_qubits:
            raise ValueError(f"Circuit has {circuit.n_qubits} qubits, which is more than simulator qubits.")
        _check_int_type("sampling shots", shots)
        _check_value_should_not_less("sampling shots", 1, shots)
        if circuit.parameterized:
            if pr is None:
                raise ValueError("Sampling a parameterized circuit need a ParameterResolver")
            if not isinstance(pr, (dict, ParameterResolver)):
                raise TypeError(f"pr requires a dict or a ParameterResolver, but get {type(pr)}!")
            pr = ParameterResolver(pr)
        else:
            pr = ParameterResolver()
        if seed is None:
            seed = int(np.random.randint(1, 2 << 20))
        else:
            _check_seed(seed)
        res = MeasureResult()
        res.add_measure(circuit.all_measures.keys())
        sim = self
        if circuit.is_measure_end and not circuit.is_noise_circuit:
            sim = Simulator(self.backend, self.n_qubits, self.seed)
            sim.set_qs(self.get_qs())
            sim.apply_circuit(circuit.remove_measure(), pr)
            circuit = Circuit(circuit.all_measures.keys())
        samples = np.array(
            sim.sim.sampling(circuit.get_cpp_obj(), pr.get_cpp_obj(), shots, res.keys_map, seed)
        ).reshape((shots, -1))
        res.collect_data(samples)
        return res

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """
        Apply hamiltonian to a simulator, this hamiltonian can be hermitian or non hermitian.

        Note:
            The quantum state may be not a normalized quantum state after apply hamiltonian.

        Args:
            hamiltonian (Hamiltonian): the hamiltonian you want to apply.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
            >>> from mindquantum.simulator import Simulator
            >>> import scipy.sparse as sp
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_circuit(Circuit().h(0))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
            >>> ham1 = Hamiltonian(QubitOperator('Z0'))
            >>> sim.apply_hamiltonian(ham1)
            >>> sim.get_qs()
            array([ 0.70710678+0.j, -0.70710678+0.j])
            >>> sim.reset()
            >>> ham2 = Hamiltonian(sp.csr_matrix([[1, 2], [3, 4]]))
            >>> sim.apply_hamiltonian(ham2)
            >>> sim.get_qs()
            array([1.+0.j, 3.+0.j])
        """
        _check_input_type('hamiltonian', Hamiltonian, hamiltonian)
        _check_hamiltonian_qubits_number(hamiltonian, self.n_qubits)
        self.sim.apply_hamiltonian(hamiltonian.get_cpp_obj())

    def get_expectation(self, hamiltonian):
        r"""
        Get expectation of the given hamiltonian. The hamiltonian could be non hermitian.

        .. math::

            E = \left<\psi\right|H\left|\psi\right>

        Args:
            hamiltonian (Hamiltonian): The hamiltonian you want to get expectation.

        Returns:
            numbers.Number, the expectation value.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_circuit(Circuit().ry(1.2, 0))
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> sim.get_expectation(ham)
            (0.36235775447667357+0j)
        """
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError(f"hamiltonian requires a Hamiltonian, but got {type(hamiltonian)}")
        _check_hamiltonian_qubits_number(hamiltonian, self.n_qubits)
        return self.sim.get_expectation(hamiltonian.get_cpp_obj())

    def get_qs(self, ket=False):
        """
        Get current quantum state of this simulator.

        Args:
            ket (bool): Whether to return the quantum state in ket format or not.
                Default: False.

        Returns:
            numpy.ndarray, the current quantum state.

        Examples:
            >>> from mindquantum.algorithm.library import qft
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.get_qs()
            array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
        """
        if not isinstance(ket, bool):
            raise TypeError(f"ket requires a bool, but get {type(ket)}")
        state = np.array(self.sim.get_qs())
        if ket:
            return '\n'.join(ket_string(state))
        return state

    def set_qs(self, quantum_state):
        """
        Set quantum state for this simulation.

        Args:
            quantum_state (numpy.ndarray): the quantum state that you want.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('projectq', 1)
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j])
            >>> sim.set_qs(np.array([1, 1]))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
        """
        if not isinstance(quantum_state, np.ndarray):
            raise TypeError(f"quantum state must be a ndarray, but get {type(quantum_state)}")
        if len(quantum_state.shape) != 1:
            raise ValueError(f"vec requires a 1-dimensional array, but get {quantum_state.shape}")
        n_qubits = np.log2(quantum_state.shape[0])
        if n_qubits % 1 != 0:
            raise ValueError(f"vec size {quantum_state.shape[0]} is not power of 2")
        n_qubits = int(n_qubits)
        if self.n_qubits != n_qubits:
            raise ValueError(f"{n_qubits} qubits vec does not match with simulation qubits ({self.n_qubits})")
        self.sim.set_qs(quantum_state / np.sqrt(np.sum(np.abs(quantum_state) ** 2)))

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def get_expectation_with_grad(
        self,
        hams,
        circ_right,
        circ_left=None,
        simulator_left=None,
        encoder_params_name=None,
        ansatz_params_name=None,
        parallel_worker=None,
    ):
        r"""
        Get a function that return the forward value and gradient w.r.t circuit parameters.

        This method is designed to calculate the expectation and its gradient shown as below.

        .. math::

            E = \left<\varphi\right|U_l^\dagger H U_r \left|\psi\right>

        where :math:`U_l` is circ_left, :math:`U_r` is circ_right, :math:`H` is hams
        and :math:`\left|\psi\right>` is the current quantum state of this simulator,
        and :math:`\left|\varphi\right>` is the quantum state of `simulator_left`.

        Args:
            hams (Hamiltonian): The hamiltonian that need to get expectation.
            circ_right (Circuit): The :math:`U_r` circuit described above.
            circ_left (Circuit): The :math:`U_l` circuit described above. By default, this circuit
                will be none, and in this situation, :math:`U_l` will be equals to
                :math:`U_r`. Default: None.
            simulator_left (Simulator): The simulator that contains :math:`\left|\varphi\right>`. If
                None, then :math:`\left|\varphi\right>` is assumed to be equals to :math:`\left|\psi\right>`.
                Default: None.
            encoder_params_name (list[str]): To specific which parameters belongs to encoder,
                that will encoder the input data into quantum state. The encoder data
                can be a batch. Default: None.
            ansatz_params_name (list[str]): To specific which parameters belongs to ansatz,
                that will be trained during training. Default: None.
            parallel_worker (int): The parallel worker numbers. The parallel workers can handle
                batch in parallel threads. Default: None.

        Returns:
            GradOpsWrapper, a grad ops wrapper than contains information to generate this grad ops.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
            >>> from mindquantum.simulator import Simulator
            >>> circ = Circuit().ry('a', 0)
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> sim = Simulator('projectq', 1)
            >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
            >>> grad_ops(np.array([1.0]))
            (array([[0.54030231+0.j]]), array([[[-0.84147098+0.j]]]))
            >>> sim1 = Simulator('projectq', 1)
            >>> prep_circ = Circuit().h(0)
            >>> ansatz = Circuit().ry('a', 0).rz('b', 0).ry('c', 0)
            >>> sim1.apply_circuit(prep_circ)
            >>> sim2 = Simulator('projectq', 1)
            >>> ham = Hamiltonian(QubitOperator(""))
            >>> grad_ops = sim2.get_expectation_with_grad(ham, ansatz, Circuit(), simulator_left=sim1)
            >>> f, g = grad_ops(np.array([7.902762e-01, 2.139225e-04, 7.795934e-01]))
            >>> f
            array([[0.99999989-7.52279618e-05j]])
        """
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        elif not isinstance(hams, list):
            raise TypeError(f"hams requires a Hamiltonian or a list of Hamiltonian, but get {type(hams)}")
        for h_tmp in hams:
            _check_input_type("hams's element", Hamiltonian, h_tmp)
            _check_hamiltonian_qubits_number(h_tmp, self.n_qubits)
        _check_input_type("circ_right", Circuit, circ_right)
        if circ_right.is_noise_circuit:
            raise ValueError("noise circuit not support yet.")
        non_hermitian = False
        if circ_left is not None:
            _check_input_type("circ_left", Circuit, circ_left)
            if circ_left.is_noise_circuit:
                raise ValueError("noise circuit not support yet.")
            non_hermitian = True
        if simulator_left is not None:
            _check_input_type("simulator_left", Simulator, simulator_left)
            if self.backend != simulator_left.backend:
                raise ValueError(
                    "simulator_left should have the same backend as this simulator, ",
                    f"which is {self.backend}, but get {simulator_left.backend}",
                )
            if self.n_qubits != simulator_left.n_qubits:
                raise ValueError(
                    "simulator_left should have the same n_qubits as this simulator, ",
                    f"which is {self.n_qubits}, but get {simulator_left.n_qubits}",
                )
            non_hermitian = True
        if non_hermitian and simulator_left is None:
            simulator_left = self
        if circ_left is None:
            circ_left = circ_right
        if circ_left.has_measure_gate or circ_right.has_measure_gate:
            raise ValueError("circuit for variational algorithm cannot have measure gate")
        if parallel_worker is not None:
            _check_int_type("parallel_worker", parallel_worker)
        url = "https://mindspore.cn/mindquantum/docs/zh-CN/r0.7/get_gradient_of_PQC_with_mindquantum.html"
        if encoder_params_name is not None:
            warnings.warn(
                (
                    "Setting encoder_params_name is perecated from version 0.7.0, please call '.as_encoder()'"
                    " of the circuit you want to work as encoder, and do not set in this API. "
                    f"Please refer to tutorial {url}"
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            encoder_params_name_old_api = encoder_params_name
        else:
            encoder_params_name_old_api = []
        if ansatz_params_name is not None:
            warnings.warn(
                (
                    "Setting ansatz_params_name is perecated from version 0.7.0, please call '.as_ansatz()' "
                    "of the circuit you want to work as ansatz, and do not set in this API. "
                    f"Please refer to tutorial {url}"
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            ansatz_params_name_old_api = ansatz_params_name
        else:
            ansatz_params_name_old_api = []

        ansatz_params_name = circ_right.all_ansatz.keys()
        encoder_params_name = circ_right.all_encoder.keys()
        if not encoder_params_name_old_api:
            encoder_params_name_old_api = encoder_params_name
        if not ansatz_params_name_old_api:
            ansatz_params_name_old_api = ansatz_params_name
        if non_hermitian:
            for i in circ_left.all_ansatz.keys():
                if i not in ansatz_params_name:
                    ansatz_params_name.append(i)
            for i in circ_left.all_encoder.keys():
                if i not in encoder_params_name:
                    encoder_params_name.append(i)
        if set(ansatz_params_name) & set(encoder_params_name):
            raise RuntimeError("Parameter cannot be both encoder and ansatz parameter.")
        if set(encoder_params_name_old_api) != set(encoder_params_name):
            raise RuntimeError(
                "You set wrong encoder parameters. Please do not set encoder_params_name anymore, "
                "but call '.as_encoder()' of circuit that you want to work as encoder."
            )
        if set(ansatz_params_name_old_api) != set(ansatz_params_name):
            raise RuntimeError(
                "You set wrong ansatz parameters. Please do not set ansatz_params_name anymore, "
                "but call '.as_ansatz()' of circuit that you want to work as ansatz."
            )
        version = "both"
        if not ansatz_params_name:
            version = "encoder"
        if not encoder_params_name:
            version = "ansatz"

        circ_n_qubits = max(circ_left.n_qubits, circ_right.n_qubits)
        if self.n_qubits < circ_n_qubits:
            raise ValueError(f"Simulator has {self.n_qubits} qubits, but circuit has {circ_n_qubits} qubits.")

        def grad_ops(*inputs):
            if version == "both" and len(inputs) != 2:
                raise ValueError("Need two inputs!")
            if version in ("encoder", "ansatz") and len(inputs) != 1:
                raise ValueError("Need one input!")
            if version == "both":
                _check_encoder(inputs[0], len(encoder_params_name))
                _check_ansatz(inputs[1], len(ansatz_params_name))
                batch_threads, mea_threads = _thread_balance(inputs[0].shape[0], len(hams), parallel_worker)
                inputs0 = inputs[0]
                inputs1 = inputs[1]
            if version == "encoder":
                _check_encoder(inputs[0], len(encoder_params_name))
                batch_threads, mea_threads = _thread_balance(inputs[0].shape[0], len(hams), parallel_worker)
                inputs0 = inputs[0]
                inputs1 = np.array([])
            if version == "ansatz":
                _check_ansatz(inputs[0], len(ansatz_params_name))
                batch_threads, mea_threads = _thread_balance(1, len(hams), parallel_worker)
                inputs0 = np.array([[]])
                inputs1 = inputs[0]
            if non_hermitian:
                f_g1_g2 = self.sim.non_hermitian_measure_with_grad(
                    [i.get_cpp_obj() for i in hams],
                    [i.get_cpp_obj(hermitian=True) for i in hams],
                    circ_left.get_cpp_obj(),
                    circ_left.get_cpp_obj(hermitian=True),
                    circ_right.get_cpp_obj(),
                    circ_right.get_cpp_obj(hermitian=True),
                    inputs0,
                    inputs1,
                    encoder_params_name,
                    ansatz_params_name,
                    batch_threads,
                    mea_threads,
                    simulator_left.sim,
                )
            else:
                f_g1_g2 = self.sim.hermitian_measure_with_grad(
                    [i.get_cpp_obj() for i in hams],
                    circ_right.get_cpp_obj(),
                    circ_right.get_cpp_obj(hermitian=True),
                    inputs0,
                    inputs1,
                    encoder_params_name,
                    ansatz_params_name,
                    batch_threads,
                    mea_threads,
                )
            res = np.array(f_g1_g2)
            if version == 'both':
                return (
                    res[:, :, 0],
                    res[:, :, 1 : 1 + len(encoder_params_name)],  # noqa:E203
                    res[:, :, 1 + len(encoder_params_name) :],  # noqa:E203
                )  # f, g1, g2
            return res[:, :, 0], res[:, :, 1:]  # f, g

        grad_wrapper = GradOpsWrapper(
            grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name, parallel_worker
        )
        grad_str = f'{self.n_qubits} qubit' + ('' if self.n_qubits == 1 else 's')
        grad_str += f' {self.backend} VQA Operator'
        grad_wrapper.set_str(grad_str)
        return grad_wrapper


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


def _thread_balance(n_prs, n_meas, parallel_worker):
    """Thread balance."""
    if parallel_worker is None:
        parallel_worker = n_meas * n_prs
    if n_meas * n_prs <= parallel_worker:
        batch_threads = n_prs
        mea_threads = n_meas
    else:
        if n_meas < n_prs:
            batch_threads = min(n_prs, parallel_worker)
            mea_threads = min(n_meas, max(1, parallel_worker // batch_threads))
        else:
            mea_threads = min(n_meas, parallel_worker)
            batch_threads = min(n_prs, max(1, parallel_worker // mea_threads))
    return batch_threads, mea_threads


def _check_hamiltonian_qubits_number(hamiltonian, sim_qubits):
    """Check hamiltonian qubits number."""
    if hamiltonian.how_to != HowTo.ORIGIN:
        if hamiltonian.n_qubits != sim_qubits:
            raise ValueError(
                f"Hamiltonian qubits is {hamiltonian.n_qubits}, not match with simulator qubits number {sim_qubits}"
            )
    else:
        if hamiltonian.n_qubits > sim_qubits:
            raise ValueError(f"Hamiltonian qubits is {hamiltonian.n_qubits}, which is bigger than simulator qubits.")


class GradOpsWrapper:  # pylint: disable=too-many-instance-attributes
    """
    Wrapper the gradient operator that with the information that generate this gradient operator.

    Args:
        grad_ops (Union[FunctionType, MethodType])): A function or a method
            that return forward value and gradient w.r.t parameters.
        hams (Hamiltonian): The hamiltonian that generate this grad ops.
        circ_right (Circuit): The right circuit that generate this grad ops.
        circ_left (Circuit): The left circuit that generate this grad ops.
        encoder_params_name (list[str]): The encoder parameters name.
        ansatz_params_name (list[str]): The ansatz parameters name.
        parallel_worker (int): The number of parallel worker to run the batch.
    """

    def __init__(
        self, grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name, parallel_worker
    ):  # pylint: disable=too-many-arguments
        """Initialize a GradOpsWrapper object."""
        self.grad_ops = grad_ops
        self.hams = hams
        self.circ_right = circ_right
        self.circ_left = circ_left
        self.encoder_params_name = encoder_params_name
        self.ansatz_params_name = ansatz_params_name
        self.parallel_worker = parallel_worker
        self.str = ''

    def __call__(self, *args):
        """Definition of a function call operator."""
        return self.grad_ops(*args)

    def set_str(self, grad_str):
        """
        Set expression for gradient operator.

        Args:
            grad_str (str): The string of QNN operator.
        """
        self.str = grad_str


def inner_product(bra_simulator: Simulator, ket_simulator: Simulator):
    """
    Calculate the inner product of two state that in the given simulator.

    Args:
        bra_simulator (Simulator): The simulator that serve as bra state.
        ket_simulator (Simulator): The simulator that serve as ket state.

    Returns:
        numbers.Number, the inner product of two quantum state.

    Examples:
        >>> from mindquantum.core.gates import RX, RY
        >>> from mindquantum.simulator import inner_product, Simulator
        >>> bra_simulator = Simulator('projectq', 1)
        >>> bra_simulator.apply_gate(RY(1.2).on(0))
        >>> ket_simulator = Simulator('projectq', 1)
        >>> ket_simulator.apply_gate(RX(2.3).on(0))
        >>> inner_product(bra_simulator, ket_simulator)
        (0.33713923320500694-0.5153852888544989j)
    """
    _check_input_type('bra_simulator', Simulator, bra_simulator)
    _check_input_type('ket_simulator', Simulator, ket_simulator)
    if bra_simulator.n_qubits != ket_simulator.n_qubits:
        raise ValueError(
            "Two simulator should have same quantum state, "
            f"but get {bra_simulator.n_qubits} and {ket_simulator.n_qubits}."
        )
    if bra_simulator.backend != ket_simulator.backend:
        raise ValueError("The backend of two simulator should be same.")
    if bra_simulator.backend == 'projectq' and ket_simulator.backend == 'projectq':
        bra_simulator.flush()
        ket_simulator.flush()
        return mb.cpu_projectq_inner_product(bra_simulator.sim, ket_simulator.sim)
    raise ValueError(f"backend for {bra_simulator.backend} not implement.")


__all__ = ['Simulator', 'get_supported_simulator', 'GradOpsWrapper', 'inner_product']
