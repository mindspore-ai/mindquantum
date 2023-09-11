# Here provides an example of using FCI method to calculate excited state energy.

import pyscf
import pyscf.fci
from pyscf.gto import Mole

GEOMETRIES = {
    'H4_0.5':   'H 0 0 0; H 0 0 0.5; H 0 0 1.0; H 0 0 1.5',
    'H4_1.0':   'H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0',
    'H4_1.5':   'H 0 0 0; H 0 0 1.5; H 0 0 3.0; H 0 0 4.5',
    'LiH_0.5':  'Li 0 0 0; H 0 0 0.5',
    'LiH_1.5':  'Li 0 0 0; H 0 0 1.5',
    'LiH_2.5':  'Li 0 0 0; H 0 0 2.5',
    'H2_1.4':   'H 0 0 0; H 0 0 1.4',
    'BeH2_1.3': 'Be 0 0 0; H 0 0 1.3; H 0 0 -1.3',
    'H2O_1.0':  'H 0 0 -1; O 0 0 0; H 0 0 1',
}

for name, geo in GEOMETRIES.items():
    print(f'[{name}]')

    # Define molecular geometry and basis set
    mol = pyscf.M(
        atom=geo,  # in Angstrom
        basis="sto-3g",
    )

    # Perform Hartree-Fock calculation
    myhf = mol.RHF().run()

    # create an FCI solver based on the SCF object
    cisolver = pyscf.fci.FCI(myhf)
    energies, wavefunctions = cisolver.kernel(nroots=2)
    print(f'FCI ground  state energy: {energies[0]}')
    print(f'FCI excited state energy: {energies[1]}')
