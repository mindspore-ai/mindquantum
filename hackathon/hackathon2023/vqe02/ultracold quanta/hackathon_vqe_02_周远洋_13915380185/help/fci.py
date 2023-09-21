# Here provides an example of using FCI method to calculate excited state energy.

import pyscf

# Define molecular geometry and basis set
mol = pyscf.M(
    atom="H 0 0 1.5; Li 0 0 0",  # in Angstrom
    basis="sto-3g",
)

# Perform Hartree-Fock calculation
myhf = mol.RHF().run()

# create an FCI solver based on the SCF object
cisolver = pyscf.fci.FCI(myhf)
energies, wavefunctions = cisolver.kernel(nroots=2)
print("FCI ground state energy: ", energies[0])
print("FCI excited state energy: ", energies[1])
