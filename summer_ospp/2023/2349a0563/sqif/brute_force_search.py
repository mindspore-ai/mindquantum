"""
Using Brute-Froce method to get better solution than Babai's algorithm in Schnorr's algorithm.
"""

import itertools
import numpy as np


class BruteForceSearch:
    def run(self, mat_d, diff, bop):
        """The outer interface that get the disturbance.
        Args:
            mat_d: The LLL-reduced matrix.
            diff: The difference between target vector and the closest vector.
            s: The symbol which element is in {-1, 1}.
            bop: The closest vector from Schnorr's algorithm.
        Return:
            vnew: The new closest vector optimized based on `bop`.
        """
        min_energy = 1e10
        t = bop - diff
        vnew = np.zeros_like(bop)
        for distur in itertools.product((-1, 0, 1), repeat=mat_d.shape[1]):
            v = (mat_d * distur).sum(axis=1).reshape((-1, 1)) + bop
            energy = np.linalg.norm(v - t)
            if energy <= min_energy:
                min_energy = energy
                vnew = v
        return vnew

    def __call__(self, mat_d, diff, bop):
        return self.run(mat_d, diff, bop)


if __name__ == "__main__":
    mat_d = np.array(
        [[1, -4, -3],
         [-2,  1,  2],
         [2,  2,  0],
         [3, -2,  4]], np.float64)

    bf = BruteForceSearch()
    diff = np.array([0, 4, 4, 2], np.float64).reshape((-1, 1))
    s = np.array([-1, -1, -1], np.float64)
    t = np.array([0, 0, 0, 240], np.float64).reshape((-1, 1))
    bop = np.array([0, 4, 4, 242], np.float64).reshape((-1, 1))
    vnew = bf.run(mat_d, diff, bop)
    print(f"target vector `t`:\n{t.ravel().tolist()}")
    print(f"schnorr's algorithm closest vector `bop`:\n{bop.ravel().tolist()}")
    print(f"new vector `vnew`:\n{vnew.ravel().tolist()}")
    print(f"norm(bop - t) = {np.linalg.norm(bop - t):.2f}")
    print(f"norm(vnew - t) = {np.linalg.norm(vnew - t):.2f}")
