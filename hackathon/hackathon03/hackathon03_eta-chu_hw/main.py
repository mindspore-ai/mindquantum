import numpy as np
import matplotlib.pyplot as plt
import time
from mindquantum import *
from mindquantum.core.gates import RX, RY, H, X, RZ, ISWAPGate, ZZ
from mindquantum.core.circuit import Circuit


def generate_encoder():
    n_qubits = 3
    enc_layer = sum(
        [U3(f'a{i}', f'b{i}', f'c{i}', i) for i in range(n_qubits)])
    coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])
    encoder = sum(
        [add_prefix(enc_layer, f'l{i}') + coupling_layer for i in range(2)])
    return encoder, encoder.params_name


def get_ansatz():
    ansatz = Circuit()
    p = 0
    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    # Ry
    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    # center
    ansatz += RY(f"theta{p}").on(0)
    # center

    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1

    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    # Rz
    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1

    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    p += 1
    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    # Ry
    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    # center
    ansatz += RY(f"theta{p}").on(0)
    # center

    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1

    ansatz += X.on(0)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RY(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    # Rz
    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1

    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(1, 0)
    p += 1
    ansatz += X.on(0)

    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1

    ansatz += X.on(0)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)

    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(1)

    ansatz += X.on(0)
    ansatz += X.on(1)
    ansatz += RZ(f"theta{p}").on(2, [0, 1])
    p += 1
    ansatz += X.on(0)
    ansatz += X.on(1)

    return ansatz


def get_grad_op(left_state, encoder, ansatz):
    sim_left = Simulator("projectq", 3)
    sim_left.set_qs(left_state)
    circ_left = Circuit()

    ham = Hamiltonian(QubitOperator(""))

    sim_right = Simulator("projectq", 3)
    circ_right = encoder + ansatz

    grad_op = sim_right.get_expectation_with_grad(ham, circ_right, circ_left, sim_left,
                                                  encoder.params_name, ansatz.params_name)

    return grad_op


def get_all_grad_ops(final_states, encoder, ansatz):
    grad_ops = []
    for state in final_states:
        grad_op = get_grad_op(state, encoder, ansatz)
        grad_ops.append(grad_op)
    return grad_ops


def gradient_for_mse(circ_right_params, grad_ops, theta_vec):
    N = len(grad_ops)

    gradient = 0
    mse = 0
    for circ_param, grad_op in zip(circ_right_params, grad_ops):
        fidelity, grad_for_encoder, grad_for_ansatz = grad_op(np.array([circ_param]), theta_vec)
        mse += (np.squeeze(np.abs(fidelity) ** 2) - 1) ** 2
        gradient_ith = 4 * (np.squeeze(np.abs(fidelity) ** 2) - 1) * np.real(
            fidelity * (np.squeeze(grad_for_ansatz).conj()))
        gradient += gradient_ith

    return mse / N, np.squeeze(gradient / N)


def get_all_final_state(encoder, param_encoder, ansatz, param_ansatz):
    states = np.zeros(shape=(len(param_encoder), 8), dtype=complex)
    for j, i in enumerate(param_encoder):
        sim = Simulator("projectq", 3)
        sim.apply_circuit(encoder, i)
        sim.apply_circuit(ansatz, param_ansatz)

        state = sim.get_qs()
        states[j, :] = state

    return states


def get_initial_states(param_data, circ):
    final_states = []
    for i in range(len(param_data)):
        sim = Simulator("projectq", circ.n_qubits)
        sim.apply_circuit(circ, param_data[i])
        final_states.append(sim.get_qs())
    return final_states

def normal(state):
    return state/np.sqrt(np.real(np.vdot(state, state)))


if __name__ == "__main__":
    train_y = np.load("./train_y.npy", allow_pickle=True)
    train_x = np.load("./train_x.npy", allow_pickle=True)

    encoder, _ = generate_encoder()
    ansatz = get_ansatz()

    param = np.random.randn(len(ansatz.params_name))
    grad_ops = get_all_grad_ops(train_y, encoder, ansatz)

    t = time.time()
    learning_rate = 2.0
    max_iterations = 300
    ith = 0
    delta_loss = 100
    mse, grad = gradient_for_mse(train_x, grad_ops, param)
    loss = [mse]
    while ith <= max_iterations and abs(delta_loss) > 1e-7:
        param = param - learning_rate * grad
        mse, grad = gradient_for_mse(train_x, grad_ops, param)
        loss.append(mse)
        delta_loss = loss[-1] - loss[-2]
        ith += 1
        if ith == 1:
            pass
        else:
            if delta_loss < 0:
                learning_rate *= 1.05
            else:
                learning_rate /= 2

        print(ith, mse, delta_loss)

    print(time.time() - t)
    training_states = get_all_final_state(encoder, train_x, ansatz, param)
    np.save("./test_y.npy", training_states)
    acc = np.real(np.mean([np.abs(np.vdot(normal(bra), ket)) for bra, ket in zip(train_y, training_states)]))
    print(f"Acc: {acc}")