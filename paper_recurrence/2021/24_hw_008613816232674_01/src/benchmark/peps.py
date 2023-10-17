import quimb as qu
import quimb.tensor as qtn
import numpy as np
import itertools

class PEP:
    def __init__(self, Lx=3, Ly=2, Ix=2, Iy=2, bond_dim=5):
        self.peps = qtn.PEPS.rand(Lx=Lx*Ix, Ly=Ly*Iy, bond_dim=bond_dim, seed=666)
        self.Lx, self.Ly, self.Ix, self.Iy = Lx, Ly, Ix, Iy
        self.ham_nn_r = self.build_H()

    def get_ZZ(self):
        return qu.kron(qu.spin_operator('Z'), qu.spin_operator('Z'))

    def build_H(self):
        _dict = {}
        schedules = [range(x) for x in [self.Ix, self.Iy, self.Lx, self.Ly]]
        for i, j, ii, jj in itertools.product(*schedules):
            idx = i*self.Lx+ii 
            idy = j*self.Ly+jj
            _dict[(idx, idy)] = (i, j)

        def check_interior(tuple_1, tuple_2):
            return _dict[tuple_1]==_dict[tuple_2]

        def get_coeff(tuple_1, tuple_2):
            if check_interior(tuple_1, tuple_2):
                return 1
            else:
                return np.random.rand()

        def _register_H2(tuple_1, tuple_2, H2):
            H2[tuple_1, tuple_2] = get_coeff(tuple_1, tuple_2) * self.get_ZZ()

        # the default two body term
        # H2 = {None: qu.ham_heis(2)}
        H2 = {}
        # single site terms
        H1 = {}

        Tx, Ty = self.Ix*self.Lx, self.Iy*self.Ly
        for i, j in itertools.product(range(Tx), range(Ty)):
            if i<=Tx-2 and j<=Ty-2:
                _register_H2((i, j), (i, j + 1), H2)
                _register_H2((i, j), (i + 1, j), H2)
            elif i==Tx-1 and j<=Ty-2:
                _register_H2((i, j), (i, j + 1), H2)
            elif j==Ty-1 and i<=Tx-2:
                _register_H2((i, j), (i + 1, j), H2)
#             else:
#                 print(i, j, Tx, Ty)
#                 raise ValueError('any pairs should be assigned a 2-qubit operator')

            H1[i, j] = 0.5 * qu.spin_operator('X') + 0.32 * qu.spin_operator('Z')

        ham_nn_r = qtn.LocalHam2D(Tx, Ty, H2=H2, H1=H1)
        return ham_nn_r

    def run(self):

        fun = qtn.SimpleUpdate

        su = fun(
            self.peps,
            self.ham_nn_r,
            chi=64,  # boundary contraction bond dim for computing energy
            compute_energy_every=10,
            compute_energy_per_site=True,
            keep_best=True,
        )


        for tau in [0.3, 0.1, 0.03, 0.01]:
            su.evolve(100, tau=tau)


if __name__ == '__main__':
    P = PEP(Lx=3, Ly=2, Ix=3, Iy=3, bond_dim=5)
    P.run()






