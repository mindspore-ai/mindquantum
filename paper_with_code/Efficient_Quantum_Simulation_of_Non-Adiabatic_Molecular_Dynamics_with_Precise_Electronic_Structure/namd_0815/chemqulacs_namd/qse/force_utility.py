import numpy as np
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermionpyscf import run_pyscf
from pyscf import gto

import pyscf





def get_hf_fd(positions, d, basis, occupied=None, active=None):
    f = []
    for idx in range(len(positions)):
        coords = positions.copy()
        mol = pyscf.gto.Mole(atom=[['Li', tuple(coords[0:3])],['H', tuple(coords[3:6])]], basis=basis)
        mol.build()
        mf = mol.RHF().run()
        coords[idx] += d
        molfwd = pyscf.gto.Mole(atom=[['Li', tuple(coords[0:3])],['H', tuple(coords[3:6])]], basis=basis)
        molfwd.build()
        coords[idx] -= 2*d
        molbwd = pyscf.gto.Mole(atom=[['Li', tuple(coords[0:3])],['H', tuple(coords[3:6])]], basis=basis)
        molbwd.build()
        tei = (molfwd.intor('int2e', aosym=1) - molbwd.intor('int2e', aosym=1))/(2*d)
        oei = (molfwd.intor('int1e_nuc', aosym=1) + molfwd.intor('int1e_kin', aosym=1) - molbwd.intor('int1e_nuc', aosym=1) - molbwd.intor('int1e_kin', aosym=1))/(2*d)
        cst = (molfwd.energy_nuc()-molbwd.energy_nuc())/(2*d)
        mooei = of.general_basis_change(oei, mf.mo_coeff, key=(1, 0))
        motei = of.general_basis_change(tei, mf.mo_coeff, key=(1, 0, 1, 0)).transpose(0, 2, 3, 1)
        smooei, smotei = of.chem.molecular_data.spinorb_from_spatial(mooei, 0.5*motei)
        f.append(of.transforms.jordan_wigner(of.InteractionOperator(cst, smooei, smotei)))
    return f

def gradient_mo_operator(mol, mo_coeffs, hcore_mo, tei_mo, with_pulay=True):
    """
    Obtain gradient operator using codes from above

    :param mol: pyscf.Mole object for getting AO integrals
    :param mo_coeffs: AO-to-MO molecular orbital coefficients
    :param hcore_mo: Core-MO matrix (n-spatial dim x n-spatial dim)
    :param tei_mo: ERI-MO tensor (n-spatial, n-spatial, n-spatial, n-spatial)
    :return: list of N x 3 InteractionOperators where N is the number of atoms.
    """
    hcore_deriv = hcore_generator(mol)
    ovrlp_deriv = overlap_generator(mol)
    eri_deriv = eri_generator(mol)
    atmlst = range(mol.natm)
    nuc = grad_nuc(mol, atmlst=atmlst)

    atom_force_ops = []
    for k, ia in enumerate(atmlst):
        h1ao = hcore_deriv(ia)
        s1ao = ovrlp_deriv(ia)
        eriao = eri_deriv(ia)
        h1mo = np.zeros_like(h1ao)
        s1mo = np.zeros_like(s1ao)
        erimo = np.zeros_like(eriao)

        for xyz in range(3):
            # Core-MO - Hellmann-Feynman term
            h1mo[xyz] = of.general_basis_change(h1ao[xyz], mo_coeffs, key=(1, 0))

            if with_pulay:
                # X-S-MO
                s1mo[xyz] = of.general_basis_change(s1ao[xyz], mo_coeffs, key=(1, 0))
            if with_pulay:
                # one-body part of wavefunction force
                h1mo[xyz] += 0.5 * (np.einsum('pj,ip->ij', hcore_mo, s1mo[xyz]) +
                                    np.einsum('ip,jp->ij', hcore_mo, s1mo[xyz]))

            # eriao in openfermion ordering Hellmann-Feynmen term
            erimo[xyz] -= of.general_basis_change(eriao[xyz], mo_coeffs,
                                                  key=(1, 0, 1, 0)).transpose(
                (0, 2, 3, 1))

            if with_pulay:
                # two-body part of wavefunction force
                erimo[xyz] += 0.5 * (np.einsum('px,xqrs', s1mo[xyz], tei_mo) +
                                     np.einsum('qx,pxrs', s1mo[xyz], tei_mo) +
                                     np.einsum('rx,pqxs', s1mo[xyz], tei_mo) +
                                     np.einsum('sx,pqrx', s1mo[xyz], tei_mo))

            # IF you want to compute lambdas then just do the spatial part
            xyz_oei, xyz_tei = spinorb_from_spatial(h1mo[xyz], 0.5 * erimo[xyz])

            atom_force_ops.append(of.InteractionOperator(nuc[ia, xyz], xyz_oei, xyz_tei))

    return atom_force_ops


def hcore_generator(mol: gto.Mole):
    """Generator for the core deriv function

    int1e_ipkin and int1e_ipnuc take the grad with respect to each
    basis function's atomic position x, y, z and place in a matrix.
    To get the gradient with respect to a particular atom we must
    add the columns of basis functions associated
    """
    aoslices = mol.aoslice_by_atom()
    h1 = mol.intor('int1e_ipkin', comp=3)  #(0.5 \nabla | p dot p | \)
    h1 += mol.intor('int1e_ipnuc', comp=3)  #(\nabla | nuc | \)
    h1 *= -1

    def hcore_deriv(atm_id):
        _, _, p0, p1 = aoslices[atm_id]
        # this part gets the derivative with respect to the electron-nuc
        # operator. See pyscf docs for more info. (p|Grad_{Ra}Sum(M) 1/r_{e} - R_{M}|q)
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(atm_id)
        vrinv[:, p0:p1] += h1[:, p0:p1]  # add the row's that aren't zero
        return vrinv + vrinv.transpose(0, 2, 1)

    return hcore_deriv


def overlap_generator(mol: gto.Mole):
    """Generator for the overlap derivfunction

    int1e_ipovlp takes the grad of the overlap
    with respect to each basis function's positions
    """
    aoslices = mol.aoslice_by_atom()
    s1 = mol.intor('int1e_ipovlp', comp=3)  # (\nabla \| \)

    def ovlp_deriv(atm_id):
        s_r = np.zeros_like(s1)
        _, _, p0, p1 = aoslices[atm_id]
        # row-idx indexes basis function.  All basis functions not on
        # a specific atom is zero.
        s_r[:, p0:p1] = s1[:, p0:p1]
        # (\nabla \| \ ) +  (\| \nabla)
        return s_r + s_r.transpose((0, 2, 1))

    return ovlp_deriv


def eri_generator(mol: gto.Mole):
    """Using int2e_ip1 = (nabla, | , )

    Remeber: chem notation (1*,1|2*,2) -> (ij|kl)

    NOTE: Prove the following is true through integral recursions

    (nabla i,j|kl) = (j,nablai|k,l) = (k,l|nabla i,j) = (k,l|j,nabla i)
    """
    aoslices = mol.aoslice_by_atom()
    eri_3 = mol.intor("int2e_ip1", comp=3)

    def eri_deriv(atm_id):
        eri_r = np.zeros_like(eri_3)
        _, _, p0, p1 = aoslices[atm_id]
        # take only the p0:p1 rows of the first index.
        # note we leverage numpy taking over all remaining jkl indices.
        # (p1 - p0, N, N, N) are non-zero
        eri_r[:, p0:p1] = eri_3[:, p0:p1]
        eri_r[:, :, p0:p1, :, :] += np.einsum('xijkl->xjikl', eri_3[:, p0:p1])
        eri_r[:, :, :, p0:p1, :] += np.einsum('xijkl->xklij', eri_3[:, p0:p1])
        eri_r[:, :, :, :, p0:p1] += np.einsum('xijkl->xklji', eri_3[:, p0:p1])
        return eri_r

    return eri_deriv


def grad_nuc(mol: gto.Mole, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates

    courtesy of pyscf and Szabo
    '''
    gs = np.zeros((mol.natm, 3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.sqrt(np.dot(r1-r2, r1-r2))
                gs[j] -= q1 * q2 * (r2-r1) / r**3
    if atmlst is not None:
        gs = gs[atmlst]
    return gs
