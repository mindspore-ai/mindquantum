import numpy as np
from pyscf import mp

def gen_mp2_energy_mask(mol_mp2, threshold=1e-4):
    t2 = np.array(mol_mp2.t2)
    nocc, nvir = t2.shape[1],t2.shape[2]
    rmask = np.asarray(mol_mp2.ao2mo().ovov).reshape(nocc, nvir, nocc, nvir)
    rmask = rmask.transpose((0,2,1,3))
    rmask = rmask*2-rmask.transpose((0,1,3,2))
    energy_mask = np.einsum("ijab,ijab->ijab",rmask,t2)
    return abs(energy_mask)>threshold

def gen_paired_mask(mol_mp2):
    t2 = np.array(mol_mp2.t2)
    nocc, nvir = t2.shape[1],t2.shape[2]
    pairmask = np.zeros((nocc, nocc, nvir, nvir))
    for ii in range(nocc):
        for jj in range(nvir):
            pairmask[ii,ii,jj,jj] = 1
    return pairmask

def get_mp2_energy_blow_threshold(mol_mp2, mask=None, threshold=1e-4):
    if (mask is None):
        mask = gen_mp2_energy_mask(mol_mp2, threshold)
    t2 = np.array(mol_mp2.t2)
    t2 = np.einsum("ijab,ijab->ijab",t2,mask)
    eris = mol_mp2.ao2mo()
    e_corr2 = mp.mp2.energy(mol_mp2, t2, eris)
    return e_corr2 - mol_mp2.e_corr

def check_frozen(mo_energy):
    if (mo_energy[1]-mo_energy[0])>2.0:
        return 1
    else:
        return 0

def mp2_frozen_core_energy(mol_mp2,frozen=[0]):
    # E(UCCSD)-E(UCCSD-frozen)=E(MP2)-E(MP2-frozen)
    e_mp2 = mol_mp2.e_tot
    mol_mp2.frozen = frozen
    mol_mp2.run(verbose=0)
    e_mp2_frozen = mol_mp2.e_tot
    return e_mp2 - e_mp2_frozen

