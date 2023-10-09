"""
Solve the XOR linear system.
"""

import copy
from typing import List
import numpy as np


class XORLinearSystemSolver:
    """Solve the linear system.
    """

    def __init__(self, exclude_zero=True):
        self.mat_m = None              # The solve matrix
        self.mat_m_bak = None          # Back the solve matrix
        self.n = -1                    # The number of variable
        self.has_solution = True       # If there's a solution
        self.exclude_zero = exclude_zero  # Exclude the zero vector
        self.solutions = []            # The solutions.

    def _gaussian_eliminate(self):
        """Gaussian elimination method. This function will modify `self.mat_m`.
        """
        m = self.m
        n = self.n
        mat_m = self.mat_m          # Reference of self.mat_m instead of copy
        r = 0
        for c in range(n):
            t = r
            for i in range(r, m):
                if mat_m[i][c] == 1:
                    t = i
                    break
            if not mat_m[t][c]:
                continue
            for i in range(c, n+1):
                mat_m[t, i], mat_m[r, i] = mat_m[r, i], mat_m[t, i]
            for i in range(r+1, m):
                if mat_m[i][c]:
                    for j in range(c, n+1):
                        mat_m[i][j] ^= mat_m[r][j]
            r += 1

        if r < m:
            for i in range(r, m):
                if mat_m[i][n] == 1:
                    self.has_solution = False

    def _get_all_solutions(self, r: int, so: List):
        """Get all solutions.

        Args:
            r: The last row index where this line is not all zero.
            so: The current solution.
        """
        mat_m = self.mat_m
        n = self.n
        if r < 0:
            if so in self.solutions:
                return
            if not self.exclude_zero:
                self.solutions.append(copy.deepcopy(so))
            elif sum(so) > 0:
                self.solutions.append(copy.deepcopy(so))
            return
        c1 = -1
        for c in range(n):
            if mat_m[r, c]:
                c1 = c
                break
        v = mat_m[r, n]
        all_known = True
        for c in range(n-1, c1, -1):
            if so[c] == -1:
                all_known = False
                so[c] = 0
                self._get_all_solutions(r, so)
                so[c] = 1
                self._get_all_solutions(r, so)
                break
            else:
                v ^= mat_m[r, c] * so[c]
        if all_known:
            so[c1] = v
            self._get_all_solutions(r-1, so)

    def _get_last_nonzero_row(self) -> int:
        """Find the last row that is not all zero.

        Return:
            rr: The last row index where this line is not all zero.
        """
        mat_m = self.mat_m
        m = self.m
        rr = m - 1
        for r in range(m-1, -1, -1):
            if sum(mat_m[r]) > 0:
                rr = r
                break
        return rr

    def validate(self):
        """To verify all solutions. It's optional.
        """
        E = self.mat_m_bak[:, :self.n]
        t = self.mat_m_bak[:, self.n: self.n+1]
        ok = True
        for so in self.solutions:
            if not (np.mod(E @ np.array(so), 2) == t).all():
                ok = False
                break
        if ok:
            print(f"Validate pass with all {len(self.solutions)} solutions.")
        else:
            print("Validate failed, the solutions is:")
            print(self.solutions)

    def solve(self, mat_m):
        """
        Args:
        mat_m: shape=(m, n+1). The binary xor linear equations which the first n columns is
            the coefficient and the last column is target vector. All elements in `mat_m` is
            in {0, 1}.
        """
        if mat_m.max() > 1 or mat_m.min() < 0:
            print(
                f"Warn: input matrix should only include {0, 1}, it will mod 2.")
            mat_m = np.mod(mat_m, 2)
        self.mat_m = mat_m.copy()
        self.mat_m_bak = mat_m.copy()
        self.m, self.n = mat_m.shape[0], mat_m.shape[1] - 1

        self.solutions.clear()
        # t1 = time.time()
        self._gaussian_eliminate()
        # t2 = time.time()

        if self.has_solution:
            r = self._get_last_nonzero_row()
            init_solution = [-1] * self.n
            self._get_all_solutions(r, init_solution)
        # t3 = time.time()

        # print(f"t2 - t1 = {t2 - t1}, t3 - t2 = {t3 - t2}.")
        return self.solutions

    def __call__(self, mat_m):
        """Solve the linear system `mat_m`
        """
        return self.solve(mat_m)


if __name__ == "__main__":
    mat = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0]
    ])
    solver = XORLinearSystemSolver()
    roots = solver(mat)
    print(roots)
    solver.validate()
