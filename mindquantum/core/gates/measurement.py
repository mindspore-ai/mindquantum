# -*- coding: utf-8 -*-
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
"""Basic module for quantum gate."""
from collections.abc import Iterable
import numpy as np
from rich.console import Console
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.utils.type_value_check import _check_int_type
from mindquantum.utils.type_value_check import _check_value_should_not_less
from mindquantum import mqbackend as mb
from mindquantum.io.display import measure_text_drawer
from .basic import NoneParameterGate


class Measure(NoneParameterGate):
    """
    Measurement gate that measure quantum qubits.

    Args:
        name (str): The key of this measurement gate. In a quantum circuit, the
            key of different measurement gate should be unique. Default: ""

    Examples:
        >>> import numpy as np
        >>> from mindquantum import qft, Circuit
        >>> from mindquantum import Measure
        >>> from mindquantum import Simulator
        >>> circ = qft(range(2))
        >>> circ += Measure('q0').on(0)
        >>> circ += Measure().on(1)
        >>> circ
        q0: ──H────PS(π/2)─────────@────M(q0)──
                      │            │
        q1: ──────────●───────H────@────M(q1)──
        >>> sim = Simulator('projectq', circ.n_qubits)
        >>> sim.apply_circuit(Circuit().h(0).x(1, 0))
        >>> sim
        projectq simulator with 2 qubits.
        Current quantum state:
        √2/2¦00⟩
        √2/2¦11⟩
        >>> res = sim.sampling(circ, shots=2000)
        >>> res
        shots: 2000
        Keys: q1 q0│0.00   0.123       0.246        0.37       0.493       0.616
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 10│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
        {'00': 986, '10': 517, '11': 497}
        >>> sim
        projectq simulator with 2 qubits.
        Current quantum state:
        √2/2¦00⟩
        √2/2¦11⟩
        >>> sim.apply_circuit(circ[:-2])
        >>> sim
        projectq simulator with 2 qubits.
        Current quantum state:
        √2/2¦00⟩
        (√2/4-√2/4j)¦10⟩
        (√2/4+√2/4j)¦11⟩
        >>> np.abs(sim.get_qs())**2
        array([0.5 , 0.  , 0.25, 0.25])
    """
    def __init__(self, name=""):
        _check_input_type('name', str, name)
        self.key = name
        NoneParameterGate.__init__(self, name)
        self.name = 'M'

    def get_cpp_obj(self):
        out = mb.get_measure_gate(self.key)
        out.obj_qubits = self.obj_qubits
        return out

    def __str__(self):
        info = ""
        if self.key and self.obj_qubits:
            info = f'({self.obj_qubits[0]}, key={self.key})'
        elif self.key:
            info = f'(key={self.key})'
        elif self.obj_qubits:
            info = f'({self.obj_qubits[0]})'
        return f"Measure{info}"

    def __repr__(self):
        return self.__str__()

    def on(self, obj_qubits, ctrl_qubits=None):
        """
        Apply this measurement gate on which qubit.

        Args:
            obj_qubits (int): A non negative int that referring to its index number.
            ctrl_qubits (int): Should be None for measure gate. Default: None.

        Examples:
            >>> from mindquantum import Circuit, Measure
            >>> from mindquantum import Simulator
            >>> sim = Simulator('projectq', 2)
            >>> circ = Circuit().h(0).x(1, 0)
            >>> circ += Measure('q0').on(0)
            >>> circ += Measure('q1').on(1)
            >>> circ
            q0: ──H────●────M(q0)──
                       │
            q1: ───────X────M(q1)──
            >>> res = sim.apply_circuit(circ)
            >>> res
            shots: 1
            Keys: q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
            ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                     11│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                       │
            {'11': 1}
            >>> sim
            projectq simulator with 2 qubits.
            Current quantum state:
            1¦11⟩
        """
        if ctrl_qubits is not None:
            raise ValueError("Measure gate can not have control qubit")
        if obj_qubits is None:
            raise ValueError("The object qubit of measurement can not be none")
        _check_int_type('obj_qubits', obj_qubits)
        _check_value_should_not_less('obj_qubits', 0, obj_qubits)
        new_gate = Measure(self.key)
        new_gate.obj_qubits = [obj_qubits]
        if not new_gate.key:
            new_gate.key = f'q{obj_qubits}'
        return new_gate

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if self.key == other.key:
            return True
        return False

    def hermitian(self):
        """Hermitian gate of measure return its self"""
        if not self.obj_qubits:
            raise ValueError("Measurement should apply on some qubit first.")
        return self.__class__(self.key).on(self.obj_qubits[0])

    def check_obj_qubits(self):
        if not self.obj_qubits:
            raise ValueError("Empty measure obj qubit")
        if len(self.obj_qubits) > 1:
            raise ValueError("Measure gate only apply on a single qubit")

    def define_projectq_gate(self):
        raise NotImplementedError


class MeasureResult:
    """
    Measurement result container

    Examples:
        >>> from mindquantum import qft
        >>> from mindquantum import Simulator
        >>> sim = Simulator('projectq', 2)
        >>> res = sim.sampling(qft(range(2)).measure_all(), shots=1000)
        >>> res
        shots: 1000
        Keys: q1 q0│0.00   0.065       0.131       0.196       0.261       0.326
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 01│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 10│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
        {'00': 250, '01': 235, '10': 261, '11': 254}
        >>> res.data
        {'00': 250, '01': 235, '10': 261, '11': 254}
    """
    def __init__(self):
        self.measures = []
        self.keys = []
        self.samples = np.array([])
        self.bit_string_data = {}
        self.shots = 0

    def add_measure(self, measure):
        """
        Add a measurement gate into this measurement result container. Measure key
        should be unique in this measurement result container.

        Args:
            measure (Union[Iterable, Measure]): One or more measure gates.
        """
        if not isinstance(measure, Iterable):
            measure = [measure]
        for m in measure:
            if not isinstance(m, Measure):
                raise ValueError("Measurement gates need to \
be objects of class 'Measurement' ")
        for m in measure:
            if m.key in self.keys:
                raise ValueError(f"Measure key {m.key} already defined.")
            self.measures.append(m)
            self.keys.append(m.key)

    @property
    def keys_map(self):
        return {i: j for j, i in enumerate(self.keys)}

    def collect_data(self, samples):
        """
        collect the measured bit string

        Args:
            samples (numpy.ndarray): A two dimensional (N x M) numpy array that stores
                the sampling bit string in 0 or 1, where N represents the number of shot
                times, and M represents the number of keys in this measurement container
        """
        self.samples = samples
        out = {}
        res = np.fliplr(self.samples)
        self.shots = len(self.samples)
        for s in res:
            s = ''.join([str(i) for i in s])
            if s in out:
                out[s] += 1
            else:
                out[s] = 1
        keys = sorted(list(out.keys()))
        self.bit_string_data = {key: out[key] for key in keys}

    def select_keys(self, *keys):
        """
        Select certain measurement keys from this measurement container

        Args:
            keys (tuple[str]): The key you want to select.

        Examples:
            >>> from mindquantum import Simulator
            >>> from mindquantum import qft, H
            >>> circ = qft(range(2)).measure('q0_0', 0).measure('q1_0', 1)
            >>> circ.h(0).measure('q0_1', 0)
            >>> circ
            q0: ──H────PS(π/2)─────────@────M(q0_0)────H────M(q0_1)──
                          │            │
            q1: ──────────●───────H────@────M(q1_0)──────────────────
            >>> sim = Simulator('projectq', circ.n_qubits)
            >>> res = sim.sampling(circ, shots=500)
            >>> new_res = res.select_keys('q0_1', 'q1_0')
            >>> new_res
            shots: 500
            Keys: q1_0 q0_1│0.00    0.07       0.139       0.209       0.278       0.348
            ───────────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                         00│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
                         01│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
                         10│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
                         11│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                           │
            {'00': 115, '01': 121, '10': 125, '11': 139}
        """
        for key in keys:
            if key not in self.keys:
                raise ValueError(f'{key} not in this measure result.')
        keys_map = self.keys_map
        idx = [keys_map[key] for key in keys]
        samples = self.samples[:, idx]
        res = MeasureResult()
        res.add_measure([self.measures[i] for i in idx])
        res.collect_data(samples)
        return res

    @property
    def data(self):
        """
        Get the sampling data.

        Returns:
            dict: The samping data.
        """
        return self.bit_string_data

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        from mindquantum.io.display._config import _MEA_RES_STYLE
        res = measure_text_drawer(self)
        res.append(self.data.__str__())
        s = '\n'.join(res)
        console = Console(record=True)
        if not console.is_jupyter:
            with console.capture() as capture:
                console.print(s, style=_MEA_RES_STYLE['style'])
            s = capture.get()
        return s

    def _repr_html_(self):
        """repr for jupyter notebook"""
        from mindquantum.io.display._config import _MEA_RES_STYLE
        from mindquantum.io.display._config import MEA_HTML_FORMAT
        res = measure_text_drawer(self)
        res.append(self.data.__str__())
        s = '\n'.join(res)
        console = Console(record=True)
        with console.capture() as _:
            console.print(s, style=_MEA_RES_STYLE['style'])
        s = console.export_html(code_format=MEA_HTML_FORMAT, inline_styles=True)
        return '\n'.join(s.split('\n')[1:])
