import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

def generate_molecule_data(dist, geometry):
    basis = "sto3g"
    spin = 0
    description = '{:.3f}'.format(dist)
    molecule_of = MolecularData(geometry, basis, multiplicity=2 * spin + 1, description=description)
    molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)
    molecule_of.save()
    print("%6.4f :\t%20.16f"%(dist, molecule_of.fci_energy))

def generate_H4():
    for dist in np.arange(0.5, 2.5, 0.1):
        geometry = [
            ["H", [0.0, 0.0, 0.0]],
            ["H", [dist, 0.0, 0.0]],
            ["H", (dist * 2.0, 0.0, 0.0)],
            ["H", (dist * 3.0, 0.0, 0.0)],
        ]
        generate_molecule_data(dist, geometry)

def generate_LiH():
    for dist in np.arange(1.0, 3.0, 0.1):
        geometry = [
            ["Li", [0.0, 0.0, 0.0]],
            ["H", [0.0, 0.0, dist]],
        ]
        generate_molecule_data(dist, geometry)

