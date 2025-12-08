import math
import itertools
import numpy as np
import openfermion
from copy import deepcopy
from openfermion import QubitOperator, MolecularData, taper_off_qubits, commutator
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, bravyi_kitaev, jordan_wigner

# from tequila.grouping.binary_rep import BinaryHamiltonian
# from tequila.grouping.binary_utils import binary_null_space
from tequila import QubitHamiltonian, Variable, quantumchemistry, gates, PauliString, minimize

def get_qubit_hamiltonian(mol, geometry, basis, charge=0, multiplicity=1, qubit_transf='jw'):
    '''
    Generating qubit hamiltonina of the given molecules with specified geometry, basis sets, charge and multiplicity
    Give its qubit form using specified transformation
    '''
    g = get_molecular_data(mol, geometry)
    mol = MolecularData(g, basis, multiplicity, charge)
    mol = run_pyscf(mol)

    ham = mol.get_molecular_hamiltonian()
    hamf = get_fermion_operator(ham)

    if qubit_transf == 'bk':
        hamq = bravyi_kitaev(hamf)
    elif qubit_transf == 'jw':
        hamq = jordan_wigner(hamf)
    else:
        raise(ValueError(qubit_transf, 'Unknown transformation specified'))

    return remove_complex(hamq)

def remove_complex(H : QubitOperator, tiny=1e-8):
    '''
    Removing near-zero complex coefficient
    '''
    real_h = QubitOperator.zero()
    for term, val in H.terms.items():
        if np.imag(val) < tiny:
            val = np.real(val)
        real_h += QubitOperator(term=term, coefficient=val)
    return real_h

def convert_mol_data_to_xyz_format(mol_data):
    '''
    Convert nuclear geometry list to .xyz format.
    '''

    xyz_str = ''
    for atom in mol_data:
        xyz_str += atom[0] +' ' + ' '.join([f"{coord:.10f}" for coord in atom[1]]) +'\n'

    return xyz_str

def get_molecular_data(mol, geometry, xyz_format=False):
    '''
    Generate the molecular data of the specified molecule
    '''
    if mol == 'h2':
        mol_data = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometry]]
        ]
    elif mol == 'lih':
        mol_data = [
            ['Li', [0, 0, 0]],
            ['H', [0, 0, geometry]]
        ]
    elif mol == 'h2o':
        # Giving symmetrically stretch H2O. ∠HOH = 107.6°
        # Geometry is distance between H-O
        angle = 107.6 / 2
        angle = math.radians(angle)
        x = geometry * math.sin(angle)
        y = geometry * math.cos(angle)
        mol_data = [
            ['O', [0, 0, 0]],
            ['H', [-x, y, 0]],
            ['H', [x, y, 0]]
        ]
    elif mol == 'n2':
        mol_data = [
            ['N', [0, 0, 0]],
            ['N', [0, 0, geometry]]
        ]
    elif mol == 'h4':
        # Angular stretch between two neighboring H. R = 1.738 Å fixed radius from center to each H.
        # Geometry is the angle of the circular sector H-origin-H.
        R = 1.738
        angle = math.radians(geometry/2)                
        x = R*math.cos(angle)
        y = R*math.sin(angle)
        mol_data = [
            ['H', [x, y, 0]],
            ['H', [x, -y, 0]],
            ['H', [-x, y, 0]],
            ['H', [-x, -y, 0]]
        ]
    elif mol == 'nh3':
        bondAngle = 107
        bondAngle = math.radians(bondAngle)
        cos = math.cos(bondAngle)
        sin = math.sin(bondAngle)

        # The idea is second and third vecctor dot product is cos(angle) * geometry^2.
        thirdyRatio = (cos - cos**2) / sin
        thirdxRatio = (1 - cos**2 - thirdyRatio**2) ** (1/2)
        mol_data = [
            ['H', [0.0, 0.0, geometry]],
            ['H', [0.0, sin * geometry, cos * geometry]],
            ['H', [thirdxRatio * geometry, thirdyRatio * geometry, cos * geometry]],
            ['N', [0.0, 0.0, 0.0]]
            ]       
    elif mol == 'h2x2':            
        r0 = 0.741
        mol_data = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, r0]],
            ['H', [0, 0, r0+geometry]],
            ['H', [0, 0, 2*r0+geometry]] 
        ]
    else:
        raise(ValueError(mol, 'Unknown moleucles given'))

    if xyz_format:
        return convert_mol_data_to_xyz_format(mol_data)
    else:
        return mol_data


def get_number_qubit(H : QubitOperator):
    '''
    Return the number of qubits in H
    '''
    n_qub = -1
    for pw, val in H.terms.items():
        for ps in pw:
            n_qub = max(n_qub, ps[0])

    return n_qub + 1

def largest_first(commuting_graph_complement):
    '''
    Using a n x n binary matrix A where A[i, j] = 0 means the underlying ith item commutes with the
    jth item.
    Returns a dictionary whose values contains mutually commuting indices.
    '''
    n = commuting_graph_complement.shape[0]

    rows = commuting_graph_complement.sum(axis=0)
    ind = np.argsort(rows)[::-1]
    m = commuting_graph_complement[ind,:][:,ind]
    colors = dict()
    c = np.zeros(n, dtype=int)
    k = 0 #color

    for i in range(n):
        neighbors = np.argwhere(m[i,:])
        colors_available = set(np.arange(1, k+1)) - set(c[[x[0] for x in neighbors]])
        term = ind[i]
        if not colors_available:
            k += 1
            c[i] = k
            colors[c[i]] = [term]
        else:
            c[i] = min(list(colors_available))
            colors[c[i]].append(term)

    return colors

def pauli2binvec(pws, n):
    '''
    Turning list of pauli words into list of binary vectors in form [z, x]
    '''
    binvecs = []
    for pw in pws:
        vec = np.zeros(2*n)
        for ps in pw:
            qub = ps[0]
            w = ps[1]
            if w == 'Z':
                vec[qub] = 1
            elif w == 'X':
                vec[qub + n] = 1
            else:
                vec[qub] = 1
                vec[qub + n] = 1
        binvecs.append(vec)
    return binvecs

def anticommute(a, b):
    '''
    Return the binary symplectic inner product between two binary vectors a and b.

    Return: 0 or 1, commute or anti-commute.
    '''
    dim = len(a) // 2
    re = a[:dim] @ b[dim:] + b[:dim] @ a[dim:]

    return re % 2

def get_commuting_group(H : QubitOperator):
    '''
    Get a dictionary of mutually commuting groups with terms in Hp
    '''
    n = get_number_qubit(H)

    pws = []
    pws_val = []
    for pw, val in H.terms.items():
        pws.append(pw)
        pws_val.append(val)

    binvecs = pauli2binvec(pws, n)
    tnum = len(binvecs)
    comm_matrix = np.zeros((tnum, tnum))

    for i in range(tnum):
        for j in range(i+1, tnum):
            comm_matrix[i, j] = 1 - anticommute(binvecs[i], binvecs[j])

    comm_matrix = np.identity(tnum) + comm_matrix + comm_matrix.T
    colors = largest_first(1 - comm_matrix)

    dict = {}
    for key, indices in colors.items():
        dict[key] = QubitOperator.zero()
        for idx in indices:
            dict[key] += QubitOperator(term=pws[idx], coefficient=pws_val[idx])

    return dict

# def get_qwc_unitary(H : QubitOperator):
#     '''
#     Get the unitary that transform commuting operators to qwc operators
#     '''
#     qh = QubitHamiltonian.from_openfermion(H)
#     bh = BinaryHamiltonian.init_from_qubit_hamiltonian(qh)

#     qwc, lag, sig = bh.single_qubit_form()

#     num = len(lag)
#     U = QubitOperator.identity()

#     for idx in range(num):
#         l = QubitHamiltonian.from_paulistrings(lag[idx].to_pauli_strings())
#         s = QubitHamiltonian.from_paulistrings(sig[idx].to_pauli_strings())
#         U *= 1 / 2 ** (1/2) * (l.to_openfermion() + s.to_openfermion())
#     return U

def get_zform_unitary(H_qwc : QubitOperator):
    '''
    Get the unitary that transforms qwc operators to all-z form. 
    '''
    qwc_ops = {} # dictionary of qub : x/y/z
    for pw, _ in H_qwc.terms.items():
        for ps in pw:
            qwc_ops[ps[0]] = ps[1]

    U = QubitOperator.identity()
    for qub, op in qwc_ops.items():
        if op != 'Z':
            U *= 1/2 ** (1/2) * (QubitOperator(term=op+str(qub)) + QubitOperator(term='Z'+str(qub)))

    return U

def qubit_wise_commuting(a : QubitOperator, b : QubitOperator):
    '''
    Check if a and b are qubit-wise commuting.
    assume a and b have only one term
    '''
    ps_dict = {}

    pw, _ = a.terms.copy().popitem()

    for ps in pw:
        ps_dict[ps[0]] = ps[1]

    pw, _ = b.terms.copy().popitem()
    for ps in pw:
        if ps[0] in ps_dict:
            if ps[1] != ps_dict[ps[0]]:
                return False

    return True

def get_qwc_group(H : QubitOperator):
    '''
    Return a list of qubit-wise commuting fragments of H
    '''
    # Preparing all terms in H into a list
    qubit_ops = []
    for pw, val in H.terms.items():
        qubit_ops.append(QubitOperator(term=pw, coefficient=val))
    n = len(qubit_ops)

    # Making commutation matrix
    comm_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            comm_matrix[i, j] = qubit_wise_commuting(qubit_ops[i], qubit_ops[j])

    # Compute commuting fragments
    comm_matrix = np.identity(n) + comm_matrix + comm_matrix.T
    colors = largest_first(1 - comm_matrix)

    # Collect commuting fragments into a list of QubitOperators
    qwc_list = []
    qwc_list_idx = 0
    for key, indices in colors.items():
        qwc_list.append(QubitOperator.zero())
        for idx in indices:
            qwc_list[qwc_list_idx] += qubit_ops[idx]
        qwc_list_idx += 1
    return qwc_list

def obtain_PES(molecule, bond_lengths, basis, method):

    if method.lower() not in ['ccsd', 'cisd', 'fci', 'hf']:
        raise(ValueError("Method not recognized, implemented methods are 'ccsd', 'cisd', 'fci', 'hf'."))

    gridpoints = len(bond_lengths)

    energies = np.zeros(gridpoints)

    for i in range(gridpoints):

        obtained_e = False
        nudged_geo_tries = 0

        while obtained_e == False:

            try:
                mol_data = get_molecular_data(molecule, bond_lengths[i], xyz_format=True)
                mol_data = quantumchemistry.Molecule(mol_data, basis)

                if method == 'cisd':
                    result = mol_data.compute_energy('detci', options={"detci__ex_level": 2})
                else:
                    result = mol_data.compute_energy(method)

                print("E = {} Eh".format(result))

                energies[i] = result
                obtained_e = True

            except:
                #Nudge geometry, cross fingers
                bond_lengths[i] += 0.00000042
                nudged_geo_tries += 1

            if nudged_geo_tries > 9:
                obtained_e = True
                energies[i] = np.nan
                print("Could not converge")

    return energies

# def get_bare_stabilizer(H : QubitOperator):
#     '''
#     Identify the stabilizer of H.
#     Currently admits only stabilizer with all z
#     since hf can only identifies the value of these terms
#     '''
#     n = get_number_qubit(H)
#     pws = []

#     for pw, _ in H.terms.items():
#         pws.append(pw)

#     binvecs = pauli2binvec(pws, n)
#     nullvecs = binary_null_space(np.array(binvecs))

#     stabs = []
#     for vec in nullvecs:
#         # If is all z
#         if all(vec[:n] == 0):
#             stab = QubitOperator.identity()
#             for i in range(n):
#                 if vec[n+i] == 1:
#                     stab = stab * QubitOperator('Z'+str(i))
#             stabs.append(stab)
#         else:
#             print('Stabilizer with x/y terms ignored. ')
#     return stabs

def hf_occ(n_spin_orbitals, n_electrons, qubit_transf='jw'):
    '''
    Returns the HF canonical orbital occupations.
    Assumes Aufbau filling.
    '''
    hf_state = np.zeros(n_spin_orbitals)
    hf_state[:n_electrons] = 1
    # hf_state = np.expand_dims(hf_state, 1)

    if qubit_transf == 'bk':
        bk_encoder = openfermion.bravyi_kitaev_code(n_spin_orbitals).encoder.toarray()
        return bk_encoder @ hf_state % 2
    elif qubit_transf == 'jw':
        return hf_state
    else:
        raise(ValueError("Unknown transformation specified"))

def correct_stabilizer_phase(stabs, hf_state):
    '''
    Accept a hf state in JW/BK encoding. Correct the phase of the z stabilizers.
    '''
    for idx in range(len(stabs)):
        pw, _ = stabs[idx].terms.copy().popitem()
        for ps in pw:
            if hf_state[ps[0]] == 1:
                stabs[idx] = stabs[idx] * -1
    return stabs

# def taper_hamiltonian(H : QubitOperator, n_spin_orbitals, n_electrons, qubit_transf):
#     '''
#     Taper off the H with the stabilizer in the correct phase based on hf state.
#     '''
#     stabs = get_bare_stabilizer(H)
#     hf = hf_occ(n_spin_orbitals, n_electrons, qubit_transf)
#     stabs = correct_stabilizer_phase(stabs, hf)
#     return remove_complex(taper_off_qubits(H, stabs))

# def xy_permutations(P,n_qubits):

# 	#generates 2^(k) equivalent entanglers related by x replaced by y & vise versa, while respecting y-parity.

# 	x_indices = []
# 	y_indices = []

# 	for i,P_i in enumerate(P):
# 		if P_i == 'x':
# 			x_indices.append(i)
# 		elif P_i == 'y':
#  			y_indices.append(i)

# 	flip_indices = x_indices + y_indices

# 	y_parity = len(y_indices)%2

# 	valid_y_counts = [2*n + (y_parity) for n in range(0,n_qubits) if 2*n + (y_parity) <= n_qubits]

# 	generated_terms = []

# 	for ynum in valid_y_counts:

# 		combs = itertools.combinations(flip_indices, ynum)

# 		for c in combs:

# 			generated_term = ['e']*n_qubits

# 			for index in c:
# 				generated_term[index] = 'y'

# 			for index in flip_indices:
# 				if index not in c:
# 					generated_term[index] = 'x'

# 			generated_terms.append(generated_term)

# 	return generated_terms

def zi_permutations(P,n_qubits):

	#generates 2^(n-k) equivalent entanglers related by trivial-ops replaced by z-ops & vise versa

	equivalent_set = []

	nonflip_indices = []
	for i,P_i in enumerate(P):
		if P_i == 'e' or P_i == 'z':
			nonflip_indices.append(i)

	bit_permutations = ["".join(seq) for seq in itertools.product("01", repeat=len(nonflip_indices))]

	for permutation in bit_permutations:

		generated_P = deepcopy(P)

		for i, bit in enumerate(permutation):
			if bit == '0': #maps to e
				generated_P[nonflip_indices[i]] = 'e'
			else: #maps to z
				generated_P[nonflip_indices[i]] = 'z'

		equivalent_set.append(generated_P)

	return equivalent_set

def Sort(sub_li):

    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key = lambda x: x[1], reverse=True)
    return sub_li


def generate_qubitop(P):

    #Converts Pauli representation used in gradient grouping algorithm to QubitOperator.

    pauli_str = ''
    for c in range(0,len(P)):
        if P[c] != 'e':
            pauli_str += P[c].upper() + str(c)+ ' '

    return QubitOperator(pauli_str)

def eval_meanfield_expectation(pauli_operator, mf_angles):

    terms = pauli_operator.terms
    len_mf_angles = len(mf_angles)

    phis = mf_angles[:len_mf_angles//2]
    thetas = mf_angles[len_mf_angles//2:]

    expectation = 0

    for pauli, value in terms.items():

        pauli_expectation = 1

        for single_pauli in pauli:

            idx = single_pauli[0]

            if single_pauli[1] == 'X':
                pauli_expectation *= np.sin(thetas[idx]) * np.cos(phis[idx])

            elif single_pauli[1] == 'Y':
                pauli_expectation *= np.sin(thetas[idx]) * np.sin(phis[idx])

            elif single_pauli[1] == 'Z':
                pauli_expectation *= np.cos(thetas[idx])

            else:
                raise(ValueError('Unexpected Pauli word ' + single_pauli[1]))

        pauli_expectation *= value
        expectation += pauli_expectation

    return expectation


def get_hamiltonian_flipindices(hamiltonian, n_qubits):

    #Return tuples of flip indices present in Hamiltonian.

    flip_indice_sets = []

    for term in hamiltonian.terms:

        flip_indices = [0]*n_qubits

        for i in term:

            if i[1] in ['X','Y']: #flip index
                flip_indices[i[0]] = 1

        if flip_indices not in flip_indice_sets:

            flip_indice_sets.append(flip_indices)

    return flip_indice_sets

def generate_representative(flip_indices,n_qubits):

    #generates parent with 1 y operation

    term = ['e']*n_qubits

    count = 0

    for i in range(0,n_qubits):

        if flip_indices[i] == 1:

            if count == 0:
                term[i] = 'y'
                count = 1

            else:
                term[i] = 'x'

    return term

def purge_nonentanglers(group, n_qubits): #filters out identity and 1 qubit ops.
    filtered_group = []

    for P in group:
        identities = 0

        for p_i in P:

            if p_i == 'e':
                identities += 1

        if identities <= n_qubits - 2: #minimum cutoff 2-qubit entanglers
            filtered_group.append(P)

    return filtered_group

# def generator_alg(P, n_qubits): #generates all entanglers related to P by transformations phi_1 and phi_2.

#     equivalent_set = []

#     xy_set = xy_permutations(P, n_qubits)

#     for P_i in xy_set:
#         xyze_set = zi_permutations(P_i, n_qubits)
#         equivalent_set += xyze_set

#     return purge_nonentanglers(equivalent_set, n_qubits)

def generate_QCC_gradient_groupings(hamiltonian, n_qubits, hf_occ, cutoff=0.001):

    QMF_angles = np.concatenate([np.array([0]*n_qubits), np.pi*hf_occ])

    hamiltonian_flip_indices = get_hamiltonian_flipindices(hamiltonian, n_qubits)

    gradient_groupings = []

    for flip_indices in hamiltonian_flip_indices:

        representative_entangler = generate_qubitop(generate_representative(flip_indices, n_qubits))

        pauli_commutator = commutator(hamiltonian, representative_entangler)

        gradient = abs(1j/2*eval_meanfield_expectation(pauli_commutator, QMF_angles))

        if gradient > cutoff:

            gradient_groupings.append( (flip_indices, round(gradient,4)) )

    gradient_groupings = Sort(gradient_groupings)

    return gradient_groupings


def get_QCC_entanglers(DIS, M, n_qubits, lexi_ordering=False):

    #Obtains top M entanglers in the DIS.
    #If M > number of DIS partitions, 1 entangler is generated for each of the M highest gradient partitions
    #and, continuously loop over all partitions until M entanglers have been generated (raster scan).

    #lexi_ordering - If True, orders selected entanglers lexicographically. Otherwise, ansatz is ordered by
    #raster-scanning in direction of descending gradient magnitude.

    if DIS == []:
        return []

    DIS = [G[0] for G in DIS]

    partitions = []

    for i in range(len(DIS)):
        repr = generate_representative(DIS[i], n_qubits)
        partitions.append(generator_alg(repr, n_qubits)) #Generates entangler representations for full DIS partition. Warning: this will be exponential w/ number of qubits.

    entanglers = []
    selecting = True
    i = 0
    while selecting:
        entanglers.append(partitions[i % len(partitions)].pop(0))
        i += 1
        if len(entanglers) >= M:
            selecting=False

    entanglers = [''.join(ent) for ent in entanglers]

    if lexi_ordering:
        entanglers = sorted(entanglers, key=str.lower)

    entanglers = [generate_qubitop(list(ent)) for ent in entanglers]

    return entanglers

def construct_QMF_ansatz(n_qubits):

    b = [Variable(name='beta_{}'.format(i)) for i in range(n_qubits)]
    g = [Variable(name='gamma_{}'.format(i)) for i in range(n_qubits)]

    def euler_rot(beta, gamma, q0):
        return gates.Rx(target=q0, angle=beta) + gates.Rz(target=q0, angle=gamma)

    for i in range(n_qubits):
        if i == 0:
            U = euler_rot(b[i],g[i],i)
        else:
            U += euler_rot(b[i],g[i],i)

    return U

def construct_QCC_ansatz(entanglers):
    #entanglers must be a list of OpenFermion QubitOperators
    #Returns the QCC unitary circuit ansatz

    t = [Variable(name='tau_{}'.format(i)) for i in range(len(entanglers))]

    for i in range(len(entanglers)):
        if i == 0:
            U = gates.ExpPauli(paulistring = PauliString.from_openfermion(list(list(entanglers[i].terms.keys())[0])), angle = t[i])
        else:
            U += gates.ExpPauli(paulistring = PauliString.from_openfermion(list(list(entanglers[i].terms.keys())[0])), angle = t[i])

    return U

def minimize_E_random_guesses(objective, method, tol, n):
    sample_energies = np.zeros(n)

    vars = objective.extract_variables()

    for t in range(n):

        initial_values = {v: np.random.uniform(0, 2*np.pi) for v in vars}
        result = minimize(objective=objective, method=method, initial_values=initial_values, tol=tol, silent=True)
        E_t = result.energy
        sample_energies[t] = E_t

    return min(sample_energies)

def init_qcc_params(hf_occ, variables):
    #initialize Euler angles at HF and entangler amplitudes at zero
    n_qubits = len(hf_occ)
    initial_values = {}

    for v in variables:
        index = str(v)[-1]
        if 'beta' in str(v):
            if hf_occ[int(index)] == 1:
                initial_values[v] = np.pi
            else:
                initial_values[v] = 0.0
        else:
            initial_values[v] = 0.0

    return initial_values
