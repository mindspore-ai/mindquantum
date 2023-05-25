import numpy as np

def binary_coding_from_matrix(enc_mtx,dec_mtx,odd=0):
    from openfermion.ops import BinaryCode, BinaryPolynomial
    from openfermion.transforms import linearize_decoder
    dec_binarypoly = linearize_decoder(dec_mtx)
    occ_spin = BinaryPolynomial('1') if (odd%2==1) else BinaryPolynomial()
    for ii in range(len(dec_binarypoly)):
    #for ii in (-2,):
        dec_binarypoly[ii] = occ_spin + dec_binarypoly[ii]
    return BinaryCode(enc_mtx, dec_binarypoly)

def get_swap_matrix(nmodes,iorb,jorb):
    encode_mtx = np.eye(nmodes, dtype=np.int32)
    encode_mtx[iorb,iorb]=0
    encode_mtx[jorb,jorb]=0
    encode_mtx[iorb,jorb]=1
    encode_mtx[jorb,iorb]=1
    return encode_mtx,encode_mtx.copy()

def get_swap_code(nmodes,iorb,jorb):
    enc_mtx,dec_mtx = get_swap_matrix(nmodes,iorb,jorb)
    code_swap = binary_coding_from_matrix(enc_mtx,dec_mtx)
    return code_swap

def code_reordering(code,ncode=-1):
    code_order = code.encoder.tocoo().col
    #print(code_order)
    code_sorted= np.sort(code_order.copy())
    if ncode<0:
        ncode = code_order.shape[0]
    for ii in range(ncode):
        jj = np.where(code_order==code_sorted[ii])[0][0]
        code = code * get_swap_code(ncode,ii,jj)
        tmp=code_order[ii]; code_order[ii]=code_order[jj]; code_order[jj]=tmp
    return code

def get_code(nmodes, nocc=0):
    from openfermion.transforms import binary_code_transform, \
        jordan_wigner_code, parity_code, bravyi_kitaev_code, \
        checksum_code, interleaved_code, weight_one_binary_addressing_code
    #code = checksum_code(nmodes,0)
    code = checksum_code(nmodes//2,nocc%2) + checksum_code(nmodes//2,nocc%2)
    code = interleaved_code(nmodes) * code
    code = code_reordering(code)
    #code_symm = jordan_wigner_code(nmodes-4) + checksum_code(2,0)
    #code_symm = jordan_wigner_code(nmodes-6)+checksum_code(2,0)+checksum_code(2,0)
    #code = code * code_symm
    #code_col = code.encoder.tocoo().col
    #print(np.where(code_col<nocc*2))
    return code

def gen_code_with_symmetry(mol,mol_hf):
    from pyscf import symm
    from openfermion import BinaryCode, BinaryPolynomial
    from openfermion import linearize_decoder
    nfrozen = int(np.sum(mol_hf.mo_energy<-2.0)) # frozen-core threshold
    nocc    = int(np.sum(mol_hf.mo_occ==2))
    norb    = len(mol_hf.mo_energy)
    ### get symmetry (irreduable representation) ###
    orbsymm = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                mol_hf.mo_coeff, check=False)
    list_symm = list(set(orbsymm))
    orbsymm = np.array(orbsymm,dtype=np.int32)
    #print(orbsymm, mol.irrep_id, mol.irrep_name)
    ### check hardly excited state ###
    # is_pair = 1 if mol_hf.mo_energy[-1]>2.0 else 0
    is_pair = 0
    ### set encoding matrix and decoding matrix ###
    # w=e(v) mod2, v=d(w)+b mod2
    cp_set = ()
    e = np.eye((norb-nfrozen)*2,dtype=np.int32)
    b = np.zeros( ((norb-nfrozen)*2,1), dtype=np.int32)
    b[-2] = (nocc-nfrozen)%2 # for spin-up
    ### constraint for spin-up ###
    cp_set += (e.shape[0]-2,)
    e[-2,0::2] = 1
    ### constraint for spin-total ###
    if not(is_pair): # no hardly excited state
        cp_set += (e.shape[0]-1,)
        e[-1,:] = 1
    ### constraint for all symmetry ###
    for isymm in list_symm: # loop for different irreduable representation
        if (isymm==orbsymm[-1]): continue
        isymm_orblist = np.where(orbsymm==isymm)[0] - nfrozen
        cp_set += (isymm_orblist[-1]*2+1,)
        for jj in isymm_orblist:
            if jj<0: continue
            e[isymm_orblist[-1]*2+1,jj*2] = 1
            e[isymm_orblist[-1]*2+1,jj*2+1] = 1
    ### find qubits with constant parity ###
    cp_set = np.sort(cp_set)
    nc_set = ()
    for ii in range(e.shape[0]):
        if ii not in cp_set:
            nc_set += (ii,)
    ### build encoder and decoder for QSBC  ###
    einv = np.array(np.linalg.inv(e),dtype=np.int32)%2
    b_prim = np.dot(einv,b)%2
    decoder_poly = linearize_decoder(einv[:,nc_set])
    for ii in range(e.shape[0]):
        if b_prim[ii]==1:
            decoder_poly[ii] += BinaryPolynomial('1')
    return BinaryCode(e[nc_set,:],decoder_poly)

def fermi_to_qubit_coding(fermi_ham, nmodes, nocc=0, code=None):
    from openfermion.transforms import binary_code_transform, \
        jordan_wigner_code, parity_code, bravyi_kitaev_code, \
        checksum_code, interleaved_code, weight_one_binary_addressing_code
    from mindquantum.core.operators.qubit_operator import QubitOperator as QOP1
    from mindquantum.core.operators import FermionOperator as FOP1
    from openfermion.ops import FermionOperator as FOP2
    fermi_ham2 = FOP2()
    fermi_ham2.terms = fermi_ham.terms.copy()
    if (code is None):
        code = checksum_code(nmodes//2,nocc%2) + checksum_code(nmodes//2,nocc%2)
        code = interleaved_code(nmodes) * code
    #code_col = code.encoder.tocoo().col
    #print(np.where(code_col<nocc*2))
    qubit_ham2 = binary_code_transform(fermi_ham2, code)
    qubit_ham = QOP1()
    qubit_ham.terms = qubit_ham2.terms.copy()
    return qubit_ham


