# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyscf.mcscf import casci
# from .vqeci_mindspore import VQECI, Ansatz, Backend, MindQuantumBackend
from .vqeci_mindspore_ss_noisy import VQECI, Ansatz, Backend

def print_formatstring(energies, occ_indices_lst):
    """
    energies: list of float
        list of energies of each state
    occ_indices_lst: list of list of int
        list of occupied indices of each state
    print the formatted string
    """
    """
    example:

    Calculated electronic
    state   Energy(Hartree) Excitation energy(eV)   Occupancy indicates\n
    S0      -74.964448      0.0000
    S1      -74.567104      10.8123                 1 -> 2
    S2      -74.528864      11.8528                 0 -> 2
    S3      -74.528864      11.8528                 1 -> 3
    """
    base_energy = energies[0]
    energy_diffs = [energy - base_energy for energy in energies]

    base_occ_indices = occ_indices_lst[0]
    occ_indices_diffs = [
        (
            set(base_occ_indices) - set(occ_indices),
            set(occ_indices) - set(base_occ_indices),
        )
        for occ_indices in occ_indices_lst
    ]
    occ_indices_diffs_str = [
        (
            f"{','.join(map(str,occ_indices_diff[0]))} -> {','.join(map(str,occ_indices_diff[1]))}"
            if len(occ_indices_diff[0]) >= 1
            else ""
        )
        for occ_indices_diff in occ_indices_diffs
    ]
    states_str = ["S" + str(i) for i in range(len(energies))]

    print("Calculated electronic state")
    print("state\tEnergy(Hartree)\tExcitation energy(eV)\tOccupancy indicates")
    for i in range(len(energies)):
        print(
            f"{states_str[i]}\t{energies[i]:7f}\t{energy_diffs[i]*27.2114:.4f}           \t{occ_indices_diffs_str[i]}"
        )

class VQECASCI(casci.CASCI):
    """
    VQECASCI
        Args:
            mf:
                SCF or Mole to define the problem size.
            ncas (int):
                Number of active spatial orbitals.
            nelecas (int):
                Number of active electrons.
            ncore (None):
                Number of core orbitals.
            initial_states (optional): Initial states for VQE
            trotter_number (int):
                Number of trotter decomposition. Used for ``UCCSD``.
            excitation_number (int):
                Number of excitations.
            weight_policy (str):
                Policy of weight of SSVQE.
                same : weight = 1 , 1 , 1 , 1 , ...
                base_first : weight = 1, 0.5 , 0.5 , 0.5 ...
                exponential : weight = 1, 0.5 , 0.25 , 0.125 ...
            use_singles: (bool):
                If ``True``, single-excitation gates are applied. Used for ``UCCSD``.
            singlet_excitation (bool):
                If ``True``, the ansatz will be spin symmetric. Used for ``UCCSD``.
            is_init_random (bool):
                If ``False``, initial parameters are initialized to 0s, else, initialized randomly.
            seed (int):
                Random seed.
        Return:
            None
    """

    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        ncore=None,
        initial_states=None,
        ansatz=Ansatz.UCCSD,
        trotter_number: int = 1,
        excitation_number: int = 0,
        weight_policy: str = "exponential",
        use_singles: bool = False,
        singlet_excitation: bool = False,
        is_init_random: bool = False,
        seed: int = 0,
        k: int=1,
        layers: int=1
    ):
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)
        self.fcisolver = VQECI(
            mf.mol,
            trotter_number=trotter_number,
            use_singles=use_singles,
            singlet_excitation=singlet_excitation,
            is_init_random=is_init_random,
            seed=seed,
            ansatz=ansatz,
            excitation_number=excitation_number,
            weight_policy=weight_policy,
            initial_states=initial_states,
            k=k,
            layers=layers
    
        
        )

    def print_energies(self):
        print_formatstring(self.fcisolver.energies, self.fcisolver.occ_indices_lst)