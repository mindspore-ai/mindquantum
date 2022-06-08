from scipy.linalg import eig
from functools import reduce
import numpy as np
from mindquantum.core import QubitOperator
from mindquantum.core import Hamiltonian
from mindquantum.core import Circuit, PhaseShift, XX


def plot_func(error_bar):
    length = []
    for i in range(len(error_bar)):
        length.append(len(error_bar[i]))
    max_len = max(length)
    for i in range(len(error_bar)):
        if len(error_bar[i]) < max_len:
            num_absent = max_len - len(error_bar[i])
            for j in range(num_absent):
                error_bar[i] = np.append(error_bar[i], error_bar[i][-1])
    ave = np.zeros(max_len)
    for i in range(max_len):
        for j in range(len(error_bar)):
            ave[i] = ave[i] + error_bar[j][i]
        ave[i] = ave[i] / len(error_bar)
    return ave


class VQELongRange:
    def __init__(self, n_qubits, alpha, theta):
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.theta = theta
        self.valid_task = ['nbSz', 'nnbSz', 'nnbsoSz', 'nnnbSz', 'nnnbsoSz']
        self.ansatz_map = {
            'nbSz': self.nbsz,
            'nnbSz': self.nnbsz,
            'nnbsoSz': self.nnbsosz,
            'nnnbSz': self.nnnbsz,
            'nnnbsoSz': self.nnnbsosz
        }

    def ham_long_range_ising(self):
        ham = QubitOperator()
        for i in range(self.n_qubits - 1):
            for j in range(i + 1, self.n_qubits):
                strength = 1 / (np.abs(i - j)**self.alpha)
                ham += QubitOperator(((i, 'X'), (j, 'X')),
                                     strength * np.sin(self.theta))
        for i in range(self.n_qubits):
            ham += np.cos(self.theta) * QubitOperator('Z' + f'{i}')
        return Hamiltonian(ham)

    def sz_block_nb(self, index, param_str):
        return XX(param_str).on([index, index + 1])

    def sz_block_nnb(self, index, param_str):
        return XX(param_str).on([index, index + 2])

    def sz_block_nnnb(self, index, param_str):
        return XX(param_str).on([index, index + 3])

    def nbsz(self, num_layer):
        ans = Circuit()
        params_idx = 0
        for _ in range(num_layer):
            for i in range(0, self.n_qubits - 1, 1):
                ans += self.sz_block_nb(i, f'p_{params_idx}')
                params_idx += 1
            for i in range(self.n_qubits):
                ans += PhaseShift({f'p_{params_idx}': (1 / 2)}).on(i)
                params_idx += 1
        return ans

    def nnbsz(self, num_layer):
        ans = Circuit()
        params_idx = 0
        for k in range(num_layer):
            if k % 2 == 0:
                for i in range(0, self.n_qubits - 2, 1):
                    ans += self.sz_block_nnb(i, f"p_{params_idx}")
                    params_idx += 1
                for i in range(self.n_qubits):
                    ans += PhaseShift({f"p_{params_idx}": (1 / 2)}).on(i)
                    params_idx += 1
            else:
                for i in range(0, self.n_qubits - 1, 1):
                    ans += self.Sz_block_nb(i, f"p_{params_idx}")
                    params_idx += 1
                for i in range(self.n_qubits):
                    ans += PhaseShift({f"p_{params_idx}": (1 / 2)}).on(i)
                    params_idx += 1
        return ans

    def nnbsosz(self, num_layer):
        ans = Circuit()
        params_block = []
        params_idx = 0
        num_params = 0
        c_num = []
        for k in range(num_layer):
            if (k % 2) == 0:
                num_params += (2 * self.n_qubits - 2)
                c_num.append(2)
            if (k % 2) == 1:
                num_params += (2 * self.n_qubits - 1)
                c_num.append(1)
        for i in range(num_params):
            params_block.append(f'p_{i}')
        num_cir_1 = c_num.count(2)
        num_cir_2 = c_num.count(1)
        while (num_cir_1 != 0):
            for i in range(0, self.n_qubits - 2, 1):
                ans += self.sz_block_nnb(i, params_block[params_idx])
                params_idx += 1
            for i in range(self.n_qubits):
                ans += PhaseShift({params_block[params_idx]: (1 / 2)}).on(i)
                params_idx += 1
            num_cir_1 -= 1
        while (num_cir_2 != 0):
            for i in range(0, self.n_qubits - 1, 1):
                ans += self.sz_block_nb(i, params_block[params_idx])
                params_idx += 1
            for i in range(self.n_qubits):
                ans += PhaseShift({params_block[params_idx]: (1 / 2)}).on(i)
                params_idx += 1
            num_cir_2 -= 1
        return ans

    def nnnbsz(self, num_layer):
        ans = Circuit()
        params_block = []
        params_idx = 0
        num_params = 0
        for k in range(num_layer):
            if k % 3 == 0:
                num_params += (2 * self.n_qubits - 3)
            if k % 3 == 1:
                num_params += (2 * self.n_qubits - 2)
            if k % 3 == 2:
                num_params += (2 * self.n_qubits - 1)
        for i in range(num_params):
            params_block.append(f'p_{i}')
        for k in range(num_layer):
            if k % 3 == 0:  #奇数层数
                for i in range(0, self.n_qubits - 3, 1):
                    ans += self.Sz_block_nnnb(i, params_block[params_idx])
                    params_idx += 1
                for i in range(self.n_qubits):
                    ans += PhaseShift({
                        params_block[params_idx]: (1 / 2)
                    }).on(i)
                    params_idx += 1
            if k % 3 == 1:  #偶数层数
                for i in range(0, self.n_qubits - 2, 1):
                    ans += self.Sz_block_nnb(i, params_block[params_idx])
                    params_idx += 1
                for i in range(self.n_qubits):
                    ans += PhaseShift({
                        params_block[params_idx]: (1 / 2)
                    }).on(i)
                    params_idx += 1
            if k % 3 == 2:
                for i in range(0, self.n_qubits - 1, 1):
                    ans += self.Sz_block_nb(i, params_block[params_idx])
                    params_idx += 1
                for i in range(self.n_qubits):
                    ans += PhaseShift({
                        params_block[params_idx]: (1 / 2)
                    }).on(i)
                    params_idx += 1
        return ans

    def nnnbsosz(self, num_layer):
        ans = Circuit()
        params_block = []
        params_idx = 0
        num_params = 0
        c_num = []
        for k in range(num_layer):
            if (k % 3) == 0:
                num_params += (2 * self.n_qubits - 3)
                c_num.append(3)
            if (k % 3) == 1:
                num_params += (2 * self.n_qubits - 2)
                c_num.append(2)
            if (k % 3) == 2:
                num_params += (2 * self.n_qubits - 1)
                c_num.append(1)

        for i in range(num_params):
            params_block.append(f'p_{i}')
        num_cir_1 = c_num.count(3)
        num_cir_2 = c_num.count(2)
        num_cir_3 = c_num.count(1)
        while (num_cir_1 != 0):
            for i in range(0, self.n_qubits - 3, 1):
                ans += self.Sz_block_nnnb(i, params_block[params_idx])
                params_idx += 1
            for i in range(self.n_qubits):
                ans += PhaseShift({params_block[params_idx]: (1 / 2)}).on(i)
                params_idx += 1
            num_cir_1 -= 1
        while (num_cir_2 != 0):
            for i in range(0, self.n_qubits - 2, 1):
                ans += self.Sz_block_nnb(i, params_block[params_idx])
                params_idx += 1
            for i in range(self.n_qubits):
                ans += PhaseShift({params_block[params_idx]: (1 / 2)}).on(i)
                params_idx += 1
            num_cir_2 -= 1
        while (num_cir_3 != 0):
            for i in range(0, self.n_qubits - 1, 1):
                ans += self.Sz_block_nb(i, params_block[params_idx])
                params_idx += 1
            for i in range(self.n_qubits):
                ans += PhaseShift({params_block[params_idx]: (1 / 2)}).on(i)
                params_idx += 1
            num_cir_3 -= 1
        return ans

    def ansatz(self, task_type, num_layer):
        if task_type not in self.valid_task:
            raise ValueError(
                f"task_type should be one of {self.valid_task}, but get {task_type}"
            )
        ans = self.ansatz_map[task_type](num_layer)
        return ans

    def get_state_vec(self, circ: Circuit, params):
        state = circ.get_qs(pr=params)
        return state

    def compute_fidelity(self, state, target_state):
        res = np.abs(np.vdot(state, target_state))
        return res


def all_z_operator(n_qubits):
    out = reduce(lambda x, y: x * y,
                 [QubitOperator(f'Z{i}') for i in range(n_qubits)])
    return Hamiltonian(out)


def all_s_operator(n_qubits, ops_type):
    out = QubitOperator()
    for i in range(n_qubits):
        out += QubitOperator(f'{ops_type}{i}', 0.5)
    return out.matrix()


def calc_ground_state(ham: Hamiltonian):
    m = ham.hamiltonian.matrix().toarray()
    v, s = eig(m)
    v = np.real(v)
    idx = np.argmin(v)
    state = s[:, idx]
    return v[idx], state