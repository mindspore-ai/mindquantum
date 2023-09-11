import random
from typing import List

from mindquantum.algorithm.nisq.chem.qubit_ucc_ansatz import *
assert QubitUCCAnsatz


class QubitUCCAnsatz_hijack(QubitUCCAnsatz):
    
    def __init__(self, n_qubits=None, n_electrons=None, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1):
        super().__init__(n_qubits, n_electrons, occ_orb, vir_orb, generalized, trotter_step)

    def _implement(self, n_qubits=None, n_electrons=None, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1):
        occ_indices = []
        vir_indices = []
        n_orb = 0
        n_orb_occ = 0
        n_orb_vir = 0
        if n_qubits is not None:
            if n_qubits % 2 != 0:
                raise ValueError('The total number of qubits (spin-orbitals) should be even.')
            n_orb = n_qubits // 2
        if n_electrons is not None:
            n_orb_occ = int(numpy.ceil(n_electrons / 2))
            n_orb_vir = n_orb - n_orb_occ
            occ_indices = list(range(n_orb_occ))
            vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]
        if occ_orb is not None:
            if len(set(occ_orb)) != len(occ_orb):
                raise ValueError("Indices for occupied orbitals should be unique!")
            n_orb_occ = len(occ_orb)
            occ_indices = occ_orb
        if vir_orb is not None:
            if len(set(vir_orb)) != len(vir_orb):
                raise ValueError("Indices for virtual orbitals should be unique!")
            n_orb_vir = len(vir_orb)
            vir_indices = vir_orb
        if set(occ_indices).intersection(vir_indices):
            raise ValueError("Occupied and virtual orbitals should be different!")
        indices_tot = occ_indices + vir_indices
        max_idx = 0
        if set(indices_tot):
            max_idx = max(set(indices_tot))
        n_orb = max(n_orb, max_idx)

        # Convert spatial-orbital indices to spin-orbital indices
        occ_indices_spin = []
        vir_indices_spin = []
        for i in occ_indices:
            occ_indices_spin.append(i * 2)
            occ_indices_spin.append(i * 2 + 1)
        for i in vir_indices:
            vir_indices_spin.append(i * 2)
            vir_indices_spin.append(i * 2 + 1)

        n_occ_spin = len(occ_indices_spin)
        n_vir_spin = len(vir_indices_spin)

        ansatz_circuit = Circuit()
        circ_snippets: List[Circuit] = []

        # Record the operator pool, which can be used for qubit-ADAPT-VQE or other iterative ansatz.
        generator_quccsd_doubles = []
        generator_quccsd_singles = []
        for trotter_idx in range(trotter_step):
            if trotter_idx == 0:
                singles_counter = 0
                for (p, q) in itertools.product(vir_indices_spin, occ_indices_spin):
                    coeff_s = ParameterResolver({f't_{trotter_idx}_q_s_{singles_counter}': 1})
                    q_pq = QubitExcitationOperator(((p, 1), (q, 0)), 1.0)
                    q_pq = q_pq - hermitian_conjugated(q_pq)
                    q_pq = q_pq.normal_ordered()
                    if list(q_pq.terms):
                        # The operator pool do not have to contain operators
                        # from different Trottered steps.
                        if trotter_idx == 0: generator_quccsd_singles.append(q_pq * coeff_s)
                        circ = self._single_qubit_excitation_circuit(q, p, coeff_s)
                        circ_snippets.append(circ)
                        singles_counter += 1

            doubles_counter = 0
            for pq_counter, (p_idx, q_idx) in enumerate(itertools.product(range(n_vir_spin), range(n_vir_spin))):
                if q_idx > p_idx: continue   # Only take half of the loop to avoid repeated excitations

                p = vir_indices_spin[p_idx]
                q = vir_indices_spin[q_idx]
                for rs_counter, (r_idx, s_idx) in enumerate(itertools.product(range(n_occ_spin), range(n_occ_spin))):
                    if s_idx > r_idx: continue   # Only take half of the loop to avoid repeated excitations

                    r = occ_indices_spin[r_idx]
                    s = occ_indices_spin[s_idx]
                    coeff_d = ParameterResolver({f't_{trotter_idx}_q_d_{doubles_counter}': 1})
                    q_pqrs = QubitExcitationOperator(((p, 1), (q, 1), (r, 0), (s, 0)), 1.0)
                    q_pqrs = q_pqrs - hermitian_conjugated(q_pqrs)
                    q_pqrs = q_pqrs.normal_ordered()
                    if list(q_pqrs.terms):
                        # The operator pool do not have to contain operators
                        # from different Trottered steps.
                        if trotter_idx == 0: generator_quccsd_doubles.append(q_pqrs * coeff_d)
                        circ = self._double_qubit_excitation_circuit(r, s, p, q, coeff_d)
                        circ_snippets.append(circ)
                        doubles_counter += 1

        #random.shuffle(circ_snippets)
        for cir in circ_snippets:
            ansatz_circuit += cir

        n_qubits_circuit = 0
        if list(ansatz_circuit): n_qubits_circuit = ansatz_circuit.n_qubits
        # If the ansatz's n_qubits is not set by user, use n_qubits_circuit.
        if self.n_qubits is None: self.n_qubits = n_qubits_circuit
        if self.n_qubits < n_qubits_circuit:
            raise ValueError(f'The number of qubits in the ansatz circuit {n_qubits_circuit} > input n_qubits {n_qubits}!')
        
        self._circuit = ansatz_circuit
        self.operator_pool = generator_quccsd_doubles + generator_quccsd_singles
