from src.circ import bpansatz, init_state
from src.entropy import get_rs_from_sim, s2, s_page
from src.HeisenbergModel import HeisenbergModel
from src.GDOpt import GDOpt
from mindquantum import Simulator, Circuit
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import trange


def train(heisenberg_model: HeisenbergModel,
          pqc: Circuit, steps: int, eta: float,
          list_qubits2keep: list,
          eps_theta: float = 0.05) -> tuple((list, list, list)):
    ham = heisenberg_model.ham()
    sim = Simulator('projectq', heisenberg_model.n_qubits)
    exp_with_grad = sim.get_expectation_with_grad(ham, pqc)

    def func(args):
        f, g = exp_with_grad(args)
        return f[0, 0].real

    def grad(args):
        f, g = exp_with_grad(args)
        return g[0, 0].real

    args_init = (np.random.rand(len(pqc.params_name)) * 2 * np.pi - np.pi) * eps_theta

    opt = GDOpt(func, grad, args_init, eta)

    s2_list = []
    grad_norm_list = []
    for _ in trange(steps):
        opt.one_step_opt()
        grad_norm_list.append(np.linalg.norm(opt.grad(opt.args)))
        sim_temp = Simulator('projectq', heisenberg_model.n_qubits)
        sim_temp.apply_circuit(pqc, opt.args)
        rs = get_rs_from_sim(sim_temp, list_qubits2keep)
        s2_list.append(s2(rs))
    cost_list = opt.curve[1:]
    return s2_list, cost_list, grad_norm_list


if __name__ == '__main__':
    n_qubits = 10
    p_layers = 100
    graph = nx.random_graphs.random_regular_graph(2, n_qubits)
    nx.draw(graph)
    plt.savefig('images/graph.svg')
    plt.cla()
    fig, axs = plt.subplots(3, 1, figsize=(12 / 2, 12))
    for eta_ in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
        circ = init_state(n_qubits) + bpansatz(n_qubits, p_layers)
        s2list, costlist, gradlist = train(
            HeisenbergModel(1, 1, graph=graph),
            circ,
            steps=100,
            eta=eta_,
            list_qubits2keep=[0, 1],
        )
        axs[0].plot(np.array(s2list) / (s_page(2, 6)), label=(r'$\eta=$' + f'{eta_}'))
        axs[1].plot(costlist)
        axs[2].plot(gradlist)
    axs[0].set(ylabel=r'$S_2 / S_{page}$',
               ylim=(0, 1))
    axs[1].set(ylabel=r'Energy $E$')
    axs[2].set(ylabel=r'$\Vert \nabla_\theta E \Vert$',
               xlabel='Iterations')
    axs[0].legend(loc='upper right')
    plt.savefig(f"images/n{n_qubits}-p{p_layers}.svg")
