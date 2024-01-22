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
from mindquantum.core.operators import FermionOperator
from mindquantum.algorithm.nisq import Transform
from mindquantum.device import QubitsTopology
#from mindquantum.utils.f import random_rng

try:
    from openfermion.ops import BosonOperator
except:
    raise ImportError("openfermion is NOT implemented !")


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
        np.random.seed(seed) # rng = random_rng(seed)
        J = np.random.random(self.lattice.n_edges()) * 2 - 1
        h = np.random.random(self.lattice.size()) * 2 - 1
        self.set_coupling_strength(J)
        self.set_external_field_strength(h)

    def ground_state_energy(self, sim: str = 'mqvector', *args, **kwargs):
        return ground_state_of_sum_zz(self.get_hamiltonian(), sim)


class XYZModel(ManyBodyModel):
    r"""
    The Hamiltonian for the XYZ model has the form

    $$
        H = \sum_{\langle i, j \rangle} ( J_x S^x_i S^x_j 
         + J_y S^y_i S^y_j 
         + J_z S^z_i S^z_j)
    $$

    where

        - The indices $\langle i, j \rangle$ run over pairs
          $i$ and $j$ of sites that are connected to each other
          in the grid
        - J_x/J_y/J_z are coupling parameters
        - S_x/S_y/S_z are spin operators

    """
    def __init__(self, lattice: QubitsTopology, Jx: float = -1, Jy: float = 0, Jz: float = 0):
        super().__init__(lattice)
        self.Jx, self.Jy, self.Jz = 0, 0, 0
        self.set_coupling_strength(Jx, Jy, Jz)
    
    def set_coupling_strength(self, Jx: float, Jy: float, Jz: float):
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz

    def get_hamiltonian(self):
        sorted_edges = sorted(list(self.lattice.edges_with_id()))
        sorted_site = sorted(list(self.lattice.all_qubit_id()))
        ham = QubitOperator()
        for (i, j) in sorted_edges:
            ham += QubitOperator(f"X{i} X{j}", self.Jx)
            ham += QubitOperator(f"Y{i} Y{j}", self.Jy)
            ham += QubitOperator(f"Z{i} Z{j}", self.Jz)
        return ham

    def random_configuration(self, seed: int = None):
        np.random.seed(seed)
        Jx = np.random.random() * 2 - 1
        Jy = np.random.random() * 2 - 1
        Jz = np.random.random() * 2 - 1
        self.set_coupling_strength(Jx, Jy, Jz)

    def ground_state_energy(self, sim: str = 'mqvector', *args, **kwargs):
        #return ground_state_of_sum_zz(self.get_hamiltonian(), sim)
        from openfermion.utils import eigenspectrum
        print('use `eigenspectrum` in openfermion to get the ground state energy.')
        try:
            return eigenspectrum(self.get_hamiltonian().to_openfermion())[0].real
        except:
            print('get the ground state energy by `eigenspectrum`: failed !!!')
            return None

class XXZModel(ManyBodyModel):
    r"""
    The Hamiltonian for the XXZ model has the form

    $$
        H = \sum_{\langle i, j \rangle} ( J_x S^x_i S^x_j
         + J_x S^y_i S^y_j
         + J_z S^z_i S^z_j)
    $$

    where

        - The indices $\langle i, j \rangle$ run over pairs
          $i$ and $j$ of sites that are connected to each other
          in the grid
        - J_x/J_z are coupling parameters
        - S_x/S_z are spin operators

    """
    def __init__(self, lattice: QubitsTopology, Jx: float = -1, Jz: float = 0):
        super().__init__(lattice)
        self.Jx, self.Jz = 0, 0
        self.set_coupling_strength(Jx, Jz)
    
    def set_coupling_strength(self, Jx: float, Jz: float):
        self.Jx, self.Jz = Jx, Jz

    def get_hamiltonian(self):
        sorted_edges = sorted(list(self.lattice.edges_with_id()))
        sorted_site = sorted(list(self.lattice.all_qubit_id()))
        ham = QubitOperator()
        for (i, j) in sorted_edges:
            ham += QubitOperator(f"X{i} X{j}", self.Jx)
            ham += QubitOperator(f"Y{i} Y{j}", self.Jx)  # Jy = Jx
            ham += QubitOperator(f"Z{i} Z{j}", self.Jz)
        return ham

    def random_configuration(self, seed: int = None):
        np.random.seed(seed)
        Jx = np.random.random() * 2 - 1
        Jz = np.random.random() * 2 - 1
        self.set_coupling_strength(Jx, Jz)

    def ground_state_energy(self, sim: str = 'mqvector', *args, **kwargs):
        #return ground_state_of_sum_zz(self.get_hamiltonian(), sim)
        from openfermion.utils import eigenspectrum
        print('use `eigenspectrum` in openfermion to get the ground state energy.')
        try:
            return eigenspectrum(self.get_hamiltonian().to_openfermion())[0].real
        except:
            print('get the ground state energy by `eigenspectrum`: failed !!!')
            return None

class FermionHubbardModel(ManyBodyModel):
    r"""
    The Hamiltonian for the spinful model has the form

    $$
        \begin{align}
        H = &- t \sum_{\langle i,j \rangle} \sum_{\sigma}
                     (a^\dagger_{i, \sigma} a_{j, \sigma} +
                      a^\dagger_{j, \sigma} a_{i, \sigma})
             + U \sum_{i} a^\dagger_{i, \uparrow} a_{i, \uparrow}
                         a^\dagger_{i, \downarrow} a_{i, \downarrow}
            \\
            &- \mu \sum_i \sum_{\sigma} a^\dagger_{i, \sigma} a_{i, \sigma}
             - h \sum_i (a^\dagger_{i, \uparrow} a_{i, \uparrow} -
                       a^\dagger_{i, \downarrow} a_{i, \downarrow})
        \end{align}
    $$

    where

        - The indices $\langle i, j \rangle$ run over pairs
          $i$ and $j$ of sites that are connected to each other
          in the grid
        - $\sigma \in \{\uparrow, \downarrow\}$ is the spin
        - $t$ is the tunneling amplitude
        - $U$ is the Coulomb potential
        - $\mu$ is the chemical potential
        - $h$ is the magnetic field

    One can also construct the Hamiltonian for the spinless model, which
    has the form

    $$
        H = - t \sum_{\langle i, j \rangle} (a^\dagger_i a_j + a^\dagger_j a_i)
            + U \sum_{\langle i, j \rangle} a^\dagger_i a_i a^\dagger_j a_j
            - \mu \sum_i a_i^\dagger a_i.
    $$
    """
    def __init__(self, lattice: QubitsTopology, t: float = -1, U: float = 0, mu: float = 0, h: float = 0, spinless: bool = True):
        super().__init__(lattice)
        self.t, self.U, self.mu, self.h = 0, 0, 0, 0
        self.spinless = spinless
        self.set_coupling_strength(t, U, mu)
        
    def set_coupling_strength(self, t: float = -1, U: float = 0, mu: float = 0):
        self.t, self.U, self.mu = 0, 0, 0

    def set_external_field_strength(self, h: float = 0):
        self.h = h

    def get_hamiltonian(self):
        sorted_edges = sorted(list(self.lattice.edges_with_id()))
        sorted_site = sorted(list(self.lattice.all_qubit_id()))
        # ham = QubitOperator()
        ham_fermi = FermionOperator()
        if self.spinless:
            try:
                assert self.h == 0
            except:
                raise ValueError(f'the magnetic field parameter h={self.h} \
                        should be 0 in the spinless case!')
            for (i, j) in sorted_edges:
                ham_fermi += FermionOperator(f"{i}^ {j}", -self.t)
                ham_fermi += FermionOperator(f"{j}^ {i}", -self.t)
                ham_fermi += FermionOperator(f"{i}^ {i} {j}^ {j}", self.U)
            for i in sorted_site:
                ham_fermi += FermionOperator(f"{i}^ {i}", -self.mu)
        else:
            for (i, j) in sorted_edges:
                i0, i1 = i*2, i*2+1
                j0, j1 = j*2, j*2+1
                ham_fermi += FermionOperator(f"{i0}^ {j0}", -self.t)
                ham_fermi += FermionOperator(f"{j0}^ {i0}", -self.t)
                ham_fermi += FermionOperator(f"{i1}^ {j1}", -self.t)
                ham_fermi += FermionOperator(f"{j1}^ {i1}", -self.t)
            for i in sorted_site:
                i0, i1 = i*2, i*2+1
                ham_fermi += FermionOperator(f"{i0}^ {i0} {i1}^ {i1}", self.U)
                ham_fermi += FermionOperator(f"{i0}^ {i0}", -self.mu)
                ham_fermi += FermionOperator(f"{i1}^ {i1}", -self.mu)
                ham_fermi += FermionOperator(f"{i0}^ {i0}", -self.h)
                ham_fermi += FermionOperator(f"{i1}^ {i1}",  self.h)
        ham = Transform(ham_fermi).jordan_wigner()
        return ham

    def random_configuration(self, seed: int = None):
        np.random.seed(seed)
        t = np.random.random() * 2 - 1
        U = np.random.random() * 2 - 1
        mu= np.random.random() * 2 - 1
        h = np.random.random() * 2 - 1
        self.set_coupling_strength(t, U, mu)
        if not self.spinless:
            self.set_external_field_strength(h)

    def ground_state_energy(self, sim: str = 'mqvector', *args, **kwargs):
        #return ground_state_of_sum_zz(self.get_hamiltonian(), sim)
        from openfermion.utils import eigenspectrum
        print('use `eigenspectrum` in openfermion to get the ground state energy.')
        try:
            return eigenspectrum(self.get_hamiltonian().to_openfermion())[0].real
        except:
            print('get the ground state energy by `eigenspectrum`: failed !!!')
            return None

class BoseHubbardModel(ManyBodyModel):
    r"""
    The Hamiltonian for the Bose-Hubbard model has the form

    $$
        H = - t \sum_{\langle i, j \rangle} (b_i^\dagger b_j + b_j^\dagger b_i)
         + V \sum_{\langle i, j \rangle} b_i^\dagger b_i b_j^\dagger b_j
         + \frac{U}{2} \sum_i b_i^\dagger b_i (b_i^\dagger b_i - 1)
         - \mu \sum_i b_i^\dagger b_i.
    $$

    where

        - The indices $\langle i, j \rangle$ run over pairs
          $i$ and $j$ of nodes that are connected to each other
          in the grid
        - $t$ is the tunneling amplitude
        - $U$ is the on-site interaction potential
        - $\mu$ is the chemical potential
        - $V$ is the dipole or nearest-neighbour interaction potential
    """
    from openfermion.ops import BosonOperator
    def __init__(self, lattice: QubitsTopology, t: float = -1, V: float = 0, U: float = 0, mu: float = 0):
        super().__init__(lattice)
        self.t, self.V, self.U, self.mu = 0, 0, 0, 0
        self.set_coupling_strength(t, V, U, mu)

    def set_coupling_strength(self, t: float = -1, V: float = 0, U: float = 0, mu: float = 0):
        self.t, self.V, self.U, self.mu = t, V, U, mu

    @staticmethod
    def BosonOperator_to_QubitOperator(ham_bose):
        # TODO: not implemented
        return ham_bose

    def get_hamiltonian(self):
        sorted_edges = sorted(list(self.lattice.edges_with_id()))
        sorted_site = sorted(list(self.lattice.all_qubit_id()))
        ham_bose = BosonOperator()
        for (i, j) in sorted_edges:
            ham_bose += BosonOperator(f'{i}^ {j}', -self.t)
            ham_bose += BosonOperator(f'{j}^ {i}', -self.t)
            ham_bose += BosonOperator(f'{i}^ {i} {j}^ {j}', self.V)
        for i in sorted_site:
            ham_bose += BosonOperator(f'{i}^ {i} {i}^ {i}', self.U/2.)
            ham_bose += BosonOperator(f'{i}^ {i}', -self.U/2.)
            ham_bose += BosonOperator(f'{i}^ {i}', -self.mu)
        # TODO: openfermion BosonOperator to mindquantum QubitOperator
        ham = self.BosonOperator_to_QubitOperator(ham_bose) #just pass, ham = ham_bose
        return ham

    def random_configuration(self, seed: int = None):
        np.random.seed(seed)
        t = np.random.random() * 2 - 1
        V = np.random.random() * 2 - 1
        U = np.random.random() * 2 - 1
        mu= np.random.random() * 2 - 1
        self.set_coupling_strength(t, V, U, mu)

    def ground_state_energy(self, sim: str = 'mqvector', *args, **kwargs):
        from openfermion.utils import eigenspectrum
        print('use `eigenspectrum` in openfermion to get the ground state energy.')
        # TODO: ImportError, NOT self-consistent for BosonOperator
        try:
            return eigenspectrum(self.get_hamiltonian())[0].real
        except:
            print('get the ground state energy by `eigenspectrum`: failed !!!')
            return None


if __name__ == '__main__':
    from mindquantum.device import GridQubits

    lattice = GridQubits(3, 2)
    ising_model = IsingModel(lattice)
    ising_model.random_configuration()
    ham = ising_model.get_hamiltonian()
    ising_model.ground_state_energy(sim='mqvector')

    xxz_model = XXZModel(lattice)
    xxz_model.random_configuration()
    ham = xxz_model.get_hamiltonian()
    xxz_model.ground_state_energy(sim='mqvector')

    xyz_model = XYZModel(lattice)
    xyz_model.random_configuration()
    ham = xyz_model.get_hamiltonian()
    xyz_model.ground_state_energy(sim='mqvector')

    # spinless
    fermiHubbard = FermionHubbardModel(lattice)
    fermiHubbard.random_configuration()
    ham = fermiHubbard.get_hamiltonian()
    fermiHubbard.ground_state_energy(sim='mqvector')

    # with spin
    fermiHubbard = FermionHubbardModel(lattice, spinless=False)
    fermiHubbard.random_configuration()
    ham = fermiHubbard.get_hamiltonian()
    fermiHubbard.ground_state_energy(sim='mqvector')

    print("Bose Hubbard model:")
    boseHubbard = BoseHubbardModel(lattice)
    boseHubbard.random_configuration()
    ham = boseHubbard.get_hamiltonian() # only get an openfermion BosonOperator
    boseHubbard.ground_state_energy(sim='mqvector')
    #print(ham)


