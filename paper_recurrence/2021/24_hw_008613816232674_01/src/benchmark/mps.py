import quimb as qu
import quimb.tensor as qtn
import itertools
import numpy as np

class MPS:
#     def __init__(self, L=8, I=2, bond_dim=32, interaction_coeff=None):
    def __init__(self, n_subsystems=3, n_qubits_per_subsystems=8, bond_dim=32, interaction_coeff=None):
        # self.mps = qtn.MPS_rand_state(L=L*I, bond_dim=bond_dim)
        # L, I = n_qubits_per_subsystems, n_subsystems
        self.n_qubits_per_subsystems, self.n_subsystems = n_qubits_per_subsystems, n_subsystems
        Tx = self.n_qubits_per_subsystems*self.n_subsystems
        self.bond_dims = [bond_dim for _ in range(Tx)]
        self.ham_nn_r = self.build_H(interaction_coeff=interaction_coeff)

    def get_ZZ(self):
        return qu.kron(qu.spin_operator('Z'), qu.spin_operator('Z'))

    def build_H(self, interaction_coeff=None):
        _dict = {}
        schedules = [range(x) for x in [self.n_subsystems, self.n_qubits_per_subsystems]]
        for i, ii in itertools.product(*schedules):
            idx = i*self.n_qubits_per_subsystems+ii 
            # _dict contains which subsystem the idx is pointing to
            _dict[idx] = i

        def check_interior(idx_1, idx_2):
            return _dict[idx_1]==_dict[idx_2]

        def get_coeff(idx_1, idx_2):
            if check_interior(idx_1, idx_2):
                return 1
            else:
                if interaction_coeff is None:
                    return np.random.rand()
                else:
                    left = _dict[idx_1]
                    return interaction_coeff[left]

        def _register_H2(idx_1, idx_2, builder):
            builder[idx_1, idx_2] += get_coeff(idx_1, idx_2), 'Z', 'Z'

        Tx = self.n_qubits_per_subsystems*self.n_subsystems

        # H2 = {}
        # H1 = {}
        builder = qtn.SpinHam1D(S=1/2)
        for i in range(Tx):
            if i<=Tx-2:
                j = i + 1
                _register_H2(i, j, builder)

            # H1[i] = 0.5 * qu.spin_operator('X') + 0.32 * qu.spin_operator('Z')
            builder[i] += 0.5, 'X'
            builder[i] += 0.32, 'Z'

        # ham_nn_r = qtn.LocalHam1D(Tx, H2=H2, H1=H1)
        ham_nn_r = builder.build_mpo(Tx)

        # def phys_dim():
        #     return 2

        # ham_nn_r.phys_dim = phys_dim
        return ham_nn_r

    def run(self):
        dmrg = qtn.DMRG2(self.ham_nn_r, bond_dims=self.bond_dims, cutoffs=1e-10)
        dmrg.solve(tol=1e-6, verbosity=1)


if __name__ == '__main__':
    M = MPS(n_subsystems=3, n_qubits_per_subsystems=8)
    M.run()