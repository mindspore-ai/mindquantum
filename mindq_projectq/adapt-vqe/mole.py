from math import sin,cos,pi
import scipy
import numpy
#from numpy import array, concatenate, zeros
from openfermionpyscf import run_pyscf
from pyscf import cc, fci, gto, scf, mcscf, ao2mo, mp
from openfermion.chem import MolecularData
import os,pickle,csv

def get_H2_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
    return geometry

def get_H4_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'H'
    atom_4 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    coordinate_3 = (bond_len*2, 0.0, 0.0)
    coordinate_4 = (bond_len*3, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),(atom_3, coordinate_3), 
                (atom_4, coordinate_4)]
    return geometry

def get_H4square_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'H'
    atom_4 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, bond_len, 0.0)
    coordinate_3 = (0.0, bond_len, 0.0)
    coordinate_4 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),(atom_3, coordinate_3), 
                (atom_4, coordinate_4)]
    return geometry

def get_H6_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'H'
    atom_4 = 'H'
    atom_5 = 'H'
    atom_6 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    coordinate_3 = (bond_len*2, 0.0, 0.0)
    coordinate_4 = (bond_len*3, 0.0, 0.0)
    coordinate_5 = (bond_len*4, 0.0, 0.0)
    coordinate_6 = (bond_len*5, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),(atom_3, coordinate_3), 
                (atom_4, coordinate_4),(atom_5, coordinate_5), (atom_6, coordinate_6)]
    return geometry

def get_LiH_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'Li'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
    return geometry

def get_N2_geo(bond_len):
    atom_1 = 'N'
    atom_2 = 'N'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
    return geometry

def get_H2O_geo(bond_len):
    atom_1 = 'O'
    atom_2 = 'H'
    atom_3 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    coordinate_3 = (cos(104.45/180*pi)*bond_len, sin(104.45/180*pi)*bond_len, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2), (atom_3, coordinate_3)]
    return geometry

def get_BeH2_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'Be'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (2*bond_len, 0.0,0.0 )
    coordinate_3 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),(atom_3, coordinate_3)]
    return geometry

def get_H2S_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'S'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (2*cos(104.5/180*pi)*bond_len, 0.0,0.0 )
    coordinate_3 = (1*cos(104.5/180*pi)*bond_len, sin(104.5/180*pi)*bond_len, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),(atom_3, coordinate_3)]
    return geometry

def get_CH2_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'C'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (2*cos(104.5/180*pi)*bond_len, 0.0,0.0 )
    coordinate_3 = (1*cos(104.5/180*pi)*bond_len, sin(104.5/180*pi)*bond_len, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),(atom_3, coordinate_3)]
    return geometry

def get_CH4_geo(bond_len):
    atom_1 = 'C'
    atom_2 = 'H'
    atom_3 = 'H'
    atom_4 = 'H'
    atom_5 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    coordinate_3 = (cos(109.5/180*pi)*bond_len, sin(109.5/180*pi)*bond_len, 0.0)
    coordinate_4 = (cos(109.5/180*pi)*bond_len, cos(120/180*pi)*sin(109.5/180*pi)*bond_len, sin(120/180*pi)*sin(109.5/180*pi)*bond_len)
    coordinate_5 = (cos(109.5/180*pi)*bond_len, cos(120/180*pi)*sin(109.5/180*pi)*bond_len, -sin(120/180*pi)*sin(109.5/180*pi)*bond_len)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2), (atom_3, coordinate_3), (atom_4, coordinate_4), (atom_5, coordinate_5)]
    return geometry

def get_C2H2_geo(bond_len):
    atom_1 = 'H'
    atom_2 = 'C'
    atom_3 = 'C'
    atom_4 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (1.06, 0.0,0.0 )
    coordinate_3 = (bond_len+1.06, 0.0, 0.0)
    coordinate_4 = (bond_len+2.12, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),(atom_3, coordinate_3),(atom_4, coordinate_4)]
    return geometry

def save_fci_wfn(fci_wfn,description):
    FCI_wfn_Root="FCI_wfns"
    path = FCI_wfn_Root
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path+"/"+description):
        os.mknod(path+"/"+description)
    outfile = open(path+"/"+description, 'wb')
    print(path+"/"+description)
    pickle.dump(fci_wfn, outfile)
    outfile.close()

def read_fci_wfn(description):
    
    FCI_wfn_Root="FCI_wfns"
    if not os.path.exists(FCI_wfn_Root):
        os.mkdir(FCI_wfn_Root)
    infile = open(FCI_wfn_Root+"/"+description, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

def classical(geometry, basis, n_orb, n_ele):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = geometry
    mol.basis = basis
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()
    
    #pt = mp.MP2(mf)
    #mp2_E, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
    conv, e_hf, mo_e, mo_hf, mo_occ = scf.hf.kernel(scf.hf.SCF(mol))
    #print(mo_hf)
    mc = mcscf.CASSCF(mf, n_orb, n_ele)
    e_fci, e_ci, wfn, mo, mo_energy = mc.mc1step()
    #mc = mcscf.CASCI(mf, n_orb, n_ele)
    #mc.fcisolver = fci.solver(mol)
    #mc.natorb = 1
    #e_fci, e_ci, wfn, mo, mo_energy = mc.kernel()

    # form the one body density matrix
    rdm1 = mc.fcisolver.make_rdm1(wfn, n_orb, n_ele)
    #rdm1 = mf.make_rdm1()[0]+mf.make_rdm1()[1]
    #print(rdm1)
    # diagonalize to yield the NOs and NO occupation 
    occ, no = scipy.linalg.eigh(rdm1)
    """ print("trace of rdm1, sum of occupation numbers, and number of electrons")
    print(numpy.trace(rdm1))
    print(numpy.sum(occ))
    print(mol.nelectron) """

    # eigenvalues are sorted in ascending order so reorder
    occ = occ[::-1]
    no = no[:, ::-1]
    #print(occ)
    #print(no)
    return e_hf, e_fci, wfn, occ

if __name__ == "__main__":
    # Classical calculation sample
    hf = []
    fci = []
    bond_length = []
    mole = 'LiH'
    for step in range(12):
        bond_len = 0.9 + 0.1*step
        bond_length.append(bond_len)
        basis = 'sto-3g'
        multiplicity = 1
        charge = 0
        geometry = get_LiH_geo(bond_len)
        # Define active space
        as_start = 0
        as_end = 4
        spatial_orbital = as_end - as_start
        n_ele = 4 - as_start*2
        e_hf, e_fci, wfn, occ = classical(geometry, basis, spatial_orbital, n_ele)
        hf.append(e_hf)
        fci.append(e_fci)
        #chara = dirad_chara(occ)
        #print('r = %.9g A' %bond_len)
        #print(occ)
        #print('E(HF) = %.9g' % hf)
        #print('E(FCI) = %.9g' % e_fci)
        # Write data to .csv file
        datas = [[mole, bond_len, e_hf, e_fci]]
        with open('./data/classical.csv'.format(mole), 'a', newline='') as f:
            writer = csv.writer(f)
            for row in datas:
                writer.writerow(row)
    print(bond_length)
    print(hf)
    print(fci)