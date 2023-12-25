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

# pylint: disable=abstract-method,no-member

"""Basic module for quantum gate."""

from collections.abc import Iterable

import numpy as np
from rich.console import Console

from mindquantum import mqbackend as mb
from mindquantum.utils.string_utils import join_without_empty
from mindquantum.utils.type_value_check import _check_input_type, _check_int_type

from .basic import FunctionalGate


class Measure(FunctionalGate):
    """
    Measurement gate that measure quantum qubits.

    Args:
        name (str): The key of this measurement gate. In a quantum circuit, the
            key of different measurement gate should be unique. Default: ``''``.
        reset_to (Union[int, None]): Reset the qubit to 0 state or 1 state. If ``None``, do not reset.
                Default: ``None``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import Measure
        >>> from mindquantum.simulator import Simulator
        >>> circ = qft(range(2))
        >>> circ += Measure('q0').on(0)
        >>> circ += Measure().on(1)
        >>> circ
              ┏━━━┓ ┏━━━━━━━━━┓         ┍━━━━━━┑
        q0: ──┨ H ┠─┨ PS(π/2) ┠───────╳─┤ M q0 ├───
              ┗━━━┛ ┗━━━━┳━━━━┛       ┃ ┕━━━━━━┙
                         ┃      ┏━━━┓ ┃ ┍━━━━━━┑
        q1: ─────────────■──────┨ H ┠─╳─┤ M q1 ├───
                                ┗━━━┛   ┕━━━━━━┙
        >>> sim = Simulator('mqvector', circ.n_qubits)
        >>> sim.apply_circuit(Circuit().h(0).x(1, 0))
        >>> sim
        mqvector simulator with 2 qubits (little endian).
        Current quantum state:
        √2/2¦00⟩
        √2/2¦11⟩
        >>> res = sim.sampling(circ, shots=2000, seed=42)
        >>> res
        shots: 2000
        Keys: q1 q0│0.00   0.124       0.248       0.372       0.496       0.621
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 10│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
        {'00': 993, '10': 506, '11': 501}
        >>> sim
        mqvector simulator with 2 qubits (little endian).
        Current quantum state:
        √2/2¦00⟩
        √2/2¦11⟩
        >>> sim.apply_circuit(circ[:-2])
        >>> sim
        mqvector simulator with 2 qubits (little endian).
        Current quantum state:
        √2/2¦00⟩
        (√2/4-√2/4j)¦10⟩
        (√2/4+√2/4j)¦11⟩
        >>> np.abs(sim.get_qs())**2
        array([0.5 , 0.  , 0.25, 0.25])
    """

    def __init__(self, name='', reset_to=None):
        """Initialize a Measure object."""
        super().__init__('Measure', 1)
        _check_input_type('name', str, name)
        if reset_to is not None:
            _check_int_type('reset_to', reset_to)
            if reset_to not in [0, 1]:
                raise ValueError(f"reset_to should be 0 or 1, but get {reset_to}")
        self.key = name
        self.reset_to = reset_to

    def get_cpp_obj(self):
        """Get the underlying C++ object."""
        if self.reset_to is None:
            return mb.gate.MeasureGate(self.key, self.obj_qubits)
        return mb.gate.MeasureGate(self.key, self.obj_qubits, self.reset_to)

    def __hash__(self):
        """Hash method."""
        return hash((self.key, self.reset_to))

    def __eq__(self, other):
        """Equality comparison operator."""
        if isinstance(other, self.__class__):
            if [self.key, self.reset_to] == [other.key, other.reset_to]:
                return True
        return False

    def __extra_prop__(self):
        """Extra prop magic method."""
        return {'key': self.key, 'reset_to': self.reset_to}

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        q_s = self.__qubits_expression__()
        k_s = f"key={self.key}" if self.key else ''
        r_s = f"reset to {self.reset_to}" if self.reset_to is not None else ''
        return join_without_empty(", ", [q_s, k_s, r_s])

    def __str_in_terminal__(self):
        """Return a string representation of the object."""
        type_s = self.__type_specific_str__()
        return f"{self.name}({type_s})" if type_s else self.name

    def __str_in_circ__(self):
        """Return a string representation of the object."""
        return f"M({self.key})" if self.reset_to is None else f"M({self.key}, reset to {self.reset_to})"

    def on(self, obj_qubits, ctrl_qubits=None):
        """
        Define which qubit the gate act on and the control qubit.

        Args:
            obj_qubits (Union[int, list[int]]): measure on which qubit.
            ctrl_qubits (Union[int, list[int]]): for measurement, we can not set control qubits. Default: ``None``.

        Returns:
            Measure, a measurement gate with will defined `obj_qubits` .
        """
        new = super().on(obj_qubits, ctrl_qubits)
        if len(new.obj_qubits) != 1:
            raise ValueError("Measure gate only apply on a single qubit")
        if new.ctrl_qubits:
            raise ValueError("Measure gate cannot have control qubits.")
        if not new.key:
            new.key = f"q{new.obj_qubits[0]}"
        return new

    def hermitian(self):
        """Hermitian gate of measure return itself."""
        if not self.obj_qubits:
            raise ValueError("Measurement should apply on some qubit first.")
        return self.__class__(self.key, self.reset_to).on(self.obj_qubits[0])


class MeasureResult:
    """
    Measurement result container.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('mqvector', 2)
        >>> res = sim.sampling(qft(range(2)).measure_all(), shots=1000, seed=42)
        >>> res
        shots: 1000
        Keys: q1 q0│0.00   0.065        0.13       0.194       0.259       0.324
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 01│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 10│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 11│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
        {'00': 230, '01': 254, '10': 257, '11': 259}
        >>> res.data
        {'00': 230, '01': 254, '10': 257, '11': 259}
    """

    def __init__(self):
        """Initialize a MeasureResult object."""
        self.measures = []
        self.keys = []
        self.samples = np.array([])
        self.bit_string_data = {}
        self.shots = 0

    def add_measure(self, measure):
        """
        Add a measurement gate into this measurement result container.

        Measure key should be unique in this measurement result container.

        Args:
            measure (Union[Iterable, Measure]): One or more measure gates.
        """
        if not isinstance(measure, Iterable):
            measure = [measure]
        for meas in measure:
            if not isinstance(meas, Measure):
                raise ValueError("Measurement gates need to be objects of class 'Measurement' ")
        for meas in measure:
            if meas.key in self.keys:
                raise ValueError(f"Measure key {meas.key} already defined.")
            self.measures.append(meas)
            self.keys.append(meas.key)

    @property
    def keys_map(self):
        """Reverse mapping for the keys."""
        return {i: j for j, i in enumerate(self.keys)}

    def collect_data(self, samples):
        """
        Collect the measured bit string.

        Args:
            samples (numpy.ndarray): A two dimensional (N x M) numpy array that stores
                the sampling bit string in 0 or 1, where N represents the number of shot
                times, and M represents the number of keys in this measurement container
        """
        self.samples = samples
        out = {}
        res = np.fliplr(self.samples)
        self.shots = len(self.samples)
        for string in res:
            string = ''.join([str(i) for i in string])
            if string in out:
                out[string] += 1
            else:
                out[string] = 1
        keys = sorted(out.keys())
        self.bit_string_data = {key: out[key] for key in keys}

    def select_keys(self, *keys):
        """
        Select certain measurement keys from this measurement container.

        Args:
            keys (tuple[str]): The key you want to select.

        Examples:
            >>> from mindquantum.algorithm.library import qft
            >>> from mindquantum.core.gates import H
            >>> from mindquantum.simulator import Simulator
            >>> circ = qft(range(2)).measure('q0_0', 0).measure('q1_0', 1)
            >>> circ.h(0).measure('q0_1', 0)
            >>> circ
                  ┏━━━┓ ┏━━━━━━━━━┓         ┍━━━━━━━━┑ ┏━━━┓ ┍━━━━━━━━┑
            q0: ──┨ H ┠─┨ PS(π/2) ┠───────╳─┤ M q0_0 ├─┨ H ┠─┤ M q0_1 ├───
                  ┗━━━┛ ┗━━━━┳━━━━┛       ┃ ┕━━━━━━━━┙ ┗━━━┛ ┕━━━━━━━━┙
                             ┃      ┏━━━┓ ┃ ┍━━━━━━━━┑
            q1: ─────────────■──────┨ H ┠─╳─┤ M q1_0 ├────────────────────
                                    ┗━━━┛   ┕━━━━━━━━┙
            >>> sim = Simulator('mqvector', circ.n_qubits)
            >>> res = sim.sampling(circ, shots=500, seed=42)
            >>> new_res = res.select_keys('q0_1', 'q1_0')
            >>> new_res
            shots: 500
            Keys: q1_0 q0_1│0.00   0.068       0.136       0.204       0.272        0.34
            ───────────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                         00│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
                         01│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
                         10│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                           │
                         11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
            {'00': 127, '01': 107, '10': 136, '11': 130}
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
            dict, The sampling data.
        """
        return self.bit_string_data

    def __str__(self):
        """Return a string representation of the object."""
        return self.__repr__()

    def __repr__(self):
        """Return a string representation of the object."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.io.display import measure_text_drawer
        from mindquantum.io.display._config import _MEA_RES_STYLE

        res = measure_text_drawer(self)
        res.append(self.data.__str__())
        string = '\n'.join(res)
        console = Console(record=True)
        if not console.is_jupyter:
            with console.capture() as capture:
                console.print(string, style=_MEA_RES_STYLE)
            return capture.get()
        return string

    def _repr_html_(self):
        """Repr for jupyter notebook."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.io.display import measure_text_drawer
        from mindquantum.io.display._config import _MEA_RES_STYLE, MEA_HTML_FORMAT

        res = measure_text_drawer(self)
        res.append(str(self.data))
        string = '\n'.join(res)
        console = Console(record=True)
        with console.capture() as _:
            console.print(string, style=_MEA_RES_STYLE)
        string = console.export_html(code_format=MEA_HTML_FORMAT, inline_styles=True)
        return '\n'.join(string.split('\n')[1:])

    def svg(self, style=None):
        """
        Display current measurement result into SVG picture in jupyter notebook.

        Args:
            style (dict, str): the style to set svg style. Currently, we support
                ``'official'``. Default: ``None``.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.io.display._config import _svg_measure_config_official
        from mindquantum.io.display.measure_res_svg_drawer import SVGMeasure

        supported_style = {
            'official': _svg_measure_config_official,
        }
        if style is None:
            style = _svg_measure_config_official
        if isinstance(style, str):
            if style not in supported_style:
                raise ValueError(f"Style not found, currently we support {list(supported_style.keys())}")
            style = supported_style[style]
        return SVGMeasure(self, style)
