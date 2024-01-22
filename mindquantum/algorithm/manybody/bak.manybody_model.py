# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Base many-body model."""
from abc import ABC, abstractmethod
from numbers import Number
from typing import List, Union

import numpy as np

from mindquantum.core.operators import QubitOperator, ground_state_of_sum_zz
from mindquantum.device import QubitsTopology
from mindquantum.utils.f import random_rng


class ManyBodyModel(ABC):
    def __init__(self, lattice: QubitsTopology):
        self.lattice = lattice

    @abstractmethod
    def get_hamiltonian(self):
        raise NotImplementedError

    def get_site_num(self):
        return self.lattice.size()

    @abstractmethod
    def random_configuration(self, seed: int = None):
        raise NotImplementedError

    def ground_state_energy(self, *args, **kwargs):
        raise RuntimeWarning("Method of calculating ground state of this model is not found.")


class IsingModel(ManyBodyModel):
    """
    H = sum_ij J_ij sigma_i sigma_j + sum_i h_i sigma_i
    """

    def __init__(self, lattice: QubitsTopology, J: Union[float, List[float]] = -1, h: Union[float, List[float]] = 0):
        super().__init__(lattice)
        self.J, self.h = 0, 0
        self.set_coupling_strength(J)
        self.set_external_field_strength(h)

    def set_coupling_strength(self, J: Union[float, List[float]]):
        if not isinstance(J, Number):
            try:
                if len(J) != self.lattice.n_edges():
                    raise ValueError(
                        f"The size of J should be the number of coupling edge of lattice ({self.lattice.n_edges()})"
                    )
                self.J = J
            except TypeError:
                raise TypeError(f"J should be a number or a iterable of number, but get {type(J)}.")
        else:
            self.J = np.ones(self.lattice.n_edges()) * J

    def set_external_field_strength(self, h: Union[float, List[float]]):
        if not isinstance(h, Number):
            try:
                if len(h) != self.lattice.size():
                    raise ValueError(
                        f"The size of h should be the number of size of lattice ({self.lattice.size()}), but get {len(h)}"
                    )
                self.h = h
            except TypeError:
                raise TypeError(f"h should be a number or a iterable of number, but get {type(h)}.")
        else:
            self.h = np.ones(self.lattice.size()) * h

    def get_hamiltonian(self):
        sorted_edges = sorted(list(self.lattice.edges_with_id()))
        sorted_site = sorted(list(self.lattice.all_qubit_id()))
        ham = QubitOperator()
        for (i, j), J in zip(sorted_edges, self.J):
            ham += QubitOperator(f"Z{i} Z{j}", J)
        for i, h in zip(sorted_site, self.h):
            ham += QubitOperator(f"Z{i}", h)
        return ham

    def random_configuration(self, seed: int = None):
        rng = random_rng(seed)
        J = rng.random(self.lattice.n_edges()) * 2 - 1
        h = rng.random(self.lattice.size()) * 2 - 1
        self.set_coupling_strength(J)
        self.set_external_field_strength(h)

    def ground_state_energy(self, sim: str = 'mqvector', *args, **kwargs):
        return ground_state_of_sum_zz(self.get_hamiltonian(), sim)


class XYZModel(ManyBodyModel):
    pass


class XXZModel(ManyBodyModel):
    pass


class FermionHubbardModel(ManyBodyModel):
    pass


class BoseHubbardModel(ManyBodyModel):
    pass


if __name__ == '__main__':
    from mindquantum.device.lattice import GridQubits

    lattice = GridQubits(3, 3)
    ising_model = IsingModel(lattice)
    ham = ising_model.get_hamiltonian()
    ising_model.random_configuration()
