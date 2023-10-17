"""The implementation of quantum witnessing."""

import time
import random
from typing import List
import itertools as it
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cobyla

from mindquantum.core.operators import QubitOperator
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum import H, X, Z, RY, UN, CNOT, Circuit


def get_separable_states() -> List:
    """Get all 3-bits bi-separable states and represented by digit that
    range from 0 to 255"""
    pm1bit = list(it.product([-1, 1], repeat=2))
    pm2bit = list(it.product([-1, 1], repeat=4))
    pm3bit = np.array([np.kron(a, b) for a in pm1bit for b in pm2bit])
    str1bit = ['0', '1']
    str2bit = ['00', '01', '10', '11']
    strs1 = [a + b for a in str1bit for b in str2bit]
    strs2 = [b + a for a in str1bit for b in str2bit]
    strs3 = [b[0] + a + b[1] for a in str1bit for b in str2bit]
    strs4 = [b[1] + a + b[0] for a in str1bit for b in str2bit]
    strs3bit = [strs1, strs2, strs3, strs4]
    sep_digits = []
    for strs in strs3bit:
        for pm in pm3bit:
            # sorted to make the string in the order of: |0>, |1>,..., |7>
            pm_strs = sorted(zip(pm, strs), key=lambda x: x[1])
            # map {-1, 1} to {0, 1}
            pm, _ = zip(*pm_strs)
            pm = np.array(pm)
            pm[pm == -1] = 0  # change -1 to 0
            i = (np.array(pm) * np.array([128, 64, 32, 16, 8, 4, 2, 1])).sum()
            sep_digits.append(i)
    sep_digits = sorted(list(set(sep_digits)))
    return sep_digits


def get_overlap(fx: List, ref: List, n=3) -> float:
    """Get overlap between `fx` and `ref`.

    Args:
        fx: The input state to calculate the overlap (fidelity).
        ref: The reference state.
        n: number of qubits.

    Return:
        act: The overlap value, which is called overlap or fidelity as well.
    """
    xor = np.array(list(map(lambda x, y: x ^ y, fx, ref)))
    act = np.abs(((-1)**xor).sum()) / 2**n
    return act


def get_recognized_entangled_states():
    """Get recognized entangled states.
    """
    n = 3  # number of qubits
    ref = [0, 0, 0, 0, 0, 1, 1, 0]  # reference states
    fxs = list(it.product([0, 1], repeat=8))  # all states
    rec_ent_states = []
    for fx in fxs:
        v = get_overlap(fx, ref, n)
        if v > 0.5:
            rec_ent_states.append(
                (np.array(fx) * np.array([128, 64, 32, 16, 8, 4, 2, 1])).sum())
    return rec_ent_states


def prepare_training_data():
    """Prepare the training data.
    """
    # bi-separable states
    sep_states = get_separable_states()
    # recognized entangled states
    rec_ent_states = get_recognized_entangled_states()
    # unrecognized entangled states
    unrec_ent_states = list(
        set(range(256)) - set(sep_states) - set(rec_ent_states))
    # split the train data according to the proportion in paper.
    half_fun = lambda x: x <= 127
    x0 = list(filter(half_fun, sep_states))
    x0 += random.sample(list(filter(half_fun, unrec_ent_states)),
                        int(256 / 2 * 0.6))
    y0 = [0] * len(x0)
    x1 = rec_ent_states
    y1 = [1] * len(x1)
    x = x0 + x1
    y = y0 + y1
    return (x, y)


def int2signs(i: int) -> List:
    """Convert a integral to a 8-bits sign array, such as 11=|0000 0111>,
    map 1 to -1, map 0 to 1, so the result is [1, 1, 1, 1, 1, -1, -1, -1];

    Args:
        i: the input integral.
    Return:
        signs: the sign array corresponding to the `i`.
    """
    signs = []
    for _ in range(8):
        tmp = -1 if i % 2 else 1
        signs.insert(0, tmp)
        i = i // 2
    return signs


def get_operation(signs: List) -> List:
    """Generate the operations according to `signs`, these rules refers to
    paper: M. Rossi, M. Huber, D. Bruß, et al. Quantum hypergraph states.

    Args:
        signs: the signs of REW states, a array with 8 bits and composed of -1 or 1.

    Return:
        ops: The qubits that Z(controlled-Z) gates apply to.
    """
    ops = []
    signs = np.array(signs)
    if signs[0] == -1:
        signs = -signs
    if signs[1] == -1:
        signs[[1, 3, 5, 7]] *= -1
        ops.append((0, ))
    if signs[2] == -1:
        signs[[2, 3, 6, 7]] *= -1
        ops.append((1, ))
    if signs[4] == -1:
        signs[[4, 5, 6, 7]] *= -1
        ops.append((2, ))
    if signs[3] == -1:
        signs[[3, 7]] *= -1
        ops.append((0, 1))
    if signs[5] == -1:
        signs[[5, 7]] *= -1
        ops.append((0, 2))
    if signs[6] == -1:
        signs[[6, 7]] *= -1
        ops.append((1, 2))
    if signs[7] == -1:
        signs[7] *= -1
        ops.append((0, 1, 2))
    return ops


def prepare_states_circuit(ops: List) -> Circuit:
    """Prepare REW states circuit.

    Args:
        ops: The operations that prepare the circuit.

    Return:
        cir: The quantum circuit to prepare the state corresponding to `ops`.
    """
    cir = Circuit()
    cir += UN(H, [0, 1, 2])
    for op in ops:
        cir += Z.on(op[0], op[1:])
    return cir


def get_encoder(i: int) -> Circuit:
    """Get the encoder circuit corresponding to `i`.

    Args:
        i: The data, which means the signs of a REW state.

    Return:
        cir: The circuit to prepare the state of determined by `i`.
    """
    signs = int2signs(i)
    ops = get_operation(signs)
    cir = prepare_states_circuit(ops)
    return cir


def get_ansatz():
    """Prepare the ansatz proposed in paper: Francesco Scala et al.
    Quantum variational learning for entanglement witnessing.
    """
    ansatz = Circuit([
        RY('theta0').on(0),
        RY('theta1').on(1),
        RY('theta2').on(2),
        CNOT(1, 0),
        CNOT(2, 0),
        CNOT(2, 1),
        RY('theta3').on(0),
        RY('theta4').on(1),
        RY('theta5').on(2),
        CNOT(1, 0),
        CNOT(2, 0),
        CNOT(2, 1),
        RY('theta6').on(0),
        RY('theta7').on(1),
        RY('theta8').on(2),
    ])
    return ansatz


def get_full_circuit(i: int) -> Circuit:
    """Get the full circuit with initial state is `i`.
    """
    encoder = get_encoder(i)
    ansatz = get_ansatz()
    cir = encoder + ansatz
    cir += X(3, [0, 1, 2])
    return cir


def train_gd():
    """Train with gradient descent method."""
    n_qubits = 4  # number of qubits
    batch_size = 16  # batch size
    lr = 0.02  # learning rate
    decay = 0.99  # learning rate decay
    n_iter = 300  # number of iteration

    sim = Simulator('mqvector', n_qubits)
    ham = Hamiltonian(QubitOperator('Z3'))
    weight = np.array([0] * 9, dtype=np.float32)
    x, y = prepare_training_data()

    for _ in tqdm.tqdm(range(n_iter)):
        grad = np.zeros_like(weight)
        for _ in range(batch_size):
            xi, yi = random.choice(list(zip(x, y)))
            cir = get_full_circuit(xi)
            grad_ops = sim.get_expectation_with_grad(ham, cir)
            f, g = grad_ops(weight)
            p = np.real((1.0 + f[0, 0]) / 2.)
            g = np.real(g.ravel())
            g0 = yi / 2 / p * g
            g1 = -(1 - yi) / 2 / (1 - p) * g
            g0 = np.clip(10.0 * g0, -20, 20)
            g1 = np.clip(g1, -20, 20)
            grad += g0 + g1
        weight -= lr * grad
        lr *= decay
    return weight


def validate_and_visualize(best_pr, save_path="fig5.png"):
    """Validate the result and plot Fig. 5, which from the paper.

    Args:
        best_pr: The trained parameters.
    """
    acts = []
    for i in range(256):
        cir = get_full_circuit(i)
        qs = cir.get_qs(pr=best_pr)
        acts.append(np.abs(qs[-1]**2).sum())
    # seprable states
    sep_states = get_separable_states()
    sep_values = np.array(acts)[sep_states]
    plt.figure(figsize=(12, 5))
    plt.bar(range(256), acts, color='cyan', width=1.0)
    plt.bar(sep_states, sep_values, color='orange', width=1.0)
    plt.plot([0, 255], [0.5, 0.5], color='red')
    plt.xlabel("Hypergraph states")
    plt.ylabel("Activation")
    plt.savefig(save_path)
    print(f"The result image has been save at: {save_path}")
    plt.show()


def train_cobyla():
    """Train the network with COBYLA optimizer to optimize the parameters.
    """
    def objective(pr):
        """The objective function.
        """
        v = 0.0
        for xi, yi in zip(x, y):
            cir = get_full_circuit(xi)
            qs = cir.get_qs(pr=pr)
            pi = np.abs(qs[-1]**2).sum()
            v -= yi * np.log(pi) + (1 - yi) * np.log(1 - pi)
        return v

    x, y = prepare_training_data()
    print(f"Finish preparing data, with {len(x)} items.")
    params = np.random.uniform(size=9)
    print("Begin training, it may take several minutes on cpu.")
    best_pr = fmin_cobyla(objective, params, cons=[])
    print(f"Finish training, the best parameters are:\n{best_pr}.\n")
    return best_pr


def run():
    t1 = time.time()
    # You can replace `train_gd()` with `train_cobyla()` to use cobyla optimizer.
    best_pr = train_gd()
    validate_and_visualize(best_pr)
    t2 = time.time()
    print("It spends {:.2f} minutes.\nEND".format((t2 - t1) / 60))


if __name__ == "__main__":
    run()
