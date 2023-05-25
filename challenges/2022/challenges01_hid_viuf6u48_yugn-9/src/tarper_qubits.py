

def get_taper_stabilizers(pyscf_mol):
    from openfermion.ops.operators import QubitOperator as QOP2
    from pyscf import symm
    mol = pyscf_mol._pyscf_data['mol']
    mol.symmetry = True; mol.build()
    nocc = pyscf_mol.n_electrons //2
    orb_irrep = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, pyscf_mol._pyscf_data['scf'].mo_coeff)
    irrep_ids = mol.irrep_id

    nirrep = len(irrep_ids)
    norb   = len(orb_irrep)
    stabilizers_list = []
    # spin-up and spin-down
    s_up=''; s_down=''; c=1.0 if(nocc%2==0) else -1.0
    for ii in range(norb):
        s_up += 'Z'+ str(ii*2) + ' '
        s_down += 'Z'+ str(ii*2+1) + ' '
    stab_up = QOP2(s_up,c)
    stab_down = QOP2(s_down,c)
    stabilizers_list.append(stab_up)
    stabilizers_list.append(stab_down)
    for ii_irrep in irrep_ids[1:]:
        s =''
        for ii in range(norb):
            if (orb_irrep[ii]==ii_irrep):
                s += 'Z'+str(ii*2)+' '+'Z'+str(ii*2+1)+' '
        stab = QOP2(s,1.0)
        stabilizers_list.append(stab)
    return stabilizers_list

def taper_qubits(qubit_hamiltonian, stabilizers_list, norb=0):
    from mindquantum.core.operators.qubit_operator import QubitOperator as QOP1
    from openfermion.ops.operators import QubitOperator as QOP2
    from openfermion.transforms.repconversions import taper_off_qubits
    qubit_ham = QOP2()
    qubit_ham.terms = qubit_hamiltonian.terms.copy()
    tapered_hamiltonian, positions = \
        taper_off_qubits(operator=qubit_ham,
                        stabilizers=stabilizers_list,
                        manual_input=False,
                        #fixed_positions=[0, 3],
                        output_tapered_positions=True)
    qubit_op = QOP1()
    qubit_op.terms = tapered_hamiltonian.terms.copy()
    return qubit_op, positions



