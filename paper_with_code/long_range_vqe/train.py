import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mindquantum.simulator import Simulator
from scipy.optimize import minimize
from model import all_s_operator, all_z_operator
from model import VQELongRange
from model import calc_ground_state
from model import plot_func


def train_func(params, grad_ops, loss_history, params_history):
    f, g = grad_ops(params)
    f1, f2 = f[0]
    g1, g2 = g[0, 0, :], g[0, 1, :]
    loss = f1 + ((f2 - 1)**2)
    gradient = (g1 + 2 * (f2 - 1) * g2)
    loss_history.append(loss)
    params_history.append(params)
    loss, gradient = float(np.real(loss)), list(np.real(gradient))
    return loss, gradient


def generate_file_path_and_name(alpha, theta, num_layer, step, n_qubits,
                                data_type, task_type, base_dir):
    rela_path = f'data/alpha={alpha}/theta={theta}/{num_layer}layer/'
    file_name = f'{data_type}-run={step}_history_N={n_qubits}_alpha={alpha}_theta={theta}_{task_type}.txt'
    abs_path = os.path.join(os.path.abspath(base_dir), rela_path)
    return abs_path, file_name


def dump_file(obj, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    joblib.dump(obj, os.path.join(file_path, file_name))


class TrainModel:
    def __init__(self, n_qubits, alpha, data_dir='./'):
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.z_op = all_z_operator(n_qubits).sparse(self.n_qubits)
        self.data_dir = os.path.abspath(data_dir)
        self.all_sx = all_s_operator(self.n_qubits, 'X')
        self.all_sy = all_s_operator(self.n_qubits, 'Y')
        self.all_sz = all_s_operator(self.n_qubits, 'Z')

    def train(self,
              task_type,
              thetas,
              num_layers,
              runs,
              callback=None,
              **kwargs):
        for theta in thetas:
            theta = round(theta, 1)
            model = VQELongRange(self.n_qubits, self.alpha, theta)
            ham = model.ham_long_range_ising()
            ham.sparse(self.n_qubits)
            gv, gs = calc_ground_state(ham)
            for num_layer in num_layers:
                all_fidelity = []
                for i in runs:
                    loss_history = []
                    params_history = []
                    circ = model.ansatz(task_type, num_layer)
                    params = circ.params_name
                    if num_layer == 1:
                        params_value = np.random.rand(len(params)) * 4 * np.pi
                    if num_layer > 1:
                        file_path, file_name = generate_file_path_and_name(
                            self.alpha, theta, num_layer - 1, i, self.n_qubits,
                            'params', task_type, self.data_dir)
                        params_his = joblib.load(
                            os.path.join(file_path, file_name))
                        params_rand = np.random.rand(
                            len(params) - len(params_his[-1])) * 4 * np.pi
                        params_value = np.append(params_his[-1], params_rand)
                    sim = Simulator('projectq', self.n_qubits)
                    grad_ops = sim.get_expectation_with_grad([ham, self.z_op],
                                                             circ,
                                                             parallel_worker=2)
                    res = minimize(
                        train_func,
                        params_value,
                        args=(grad_ops, loss_history, params_history),
                        jac=True,
                        method='l-bfgs-b',
                        tol=1e-10,
                        options={
                            #    'disp': True,
                            'eps': 1e-12,
                            'maxls': 40
                        })
                    print(
                        f"theta: {theta}, num_layer: {num_layer}, runs: {i}, ground_e: {str(gv)[:15]}, trained_e: {str(res.fun)[:15]}"
                    )
                    l_filepath, l_filename = generate_file_path_and_name(
                        self.alpha, theta, num_layer, i, self.n_qubits, 'loss',
                        task_type, self.data_dir)
                    dump_file(loss_history, l_filepath, l_filename)

                    p_filepath, p_filename = generate_file_path_and_name(
                        self.alpha, theta, num_layer, i, self.n_qubits,
                        'params', task_type, self.data_dir)
                    dump_file(params_history, p_filepath, p_filename)

                    fidelity = []
                    for param in params_history:
                        final_state = model.get_state_vec(circ, param)
                        fid = model.compute_fidelity(final_state, gs)
                        fidelity.append(fid)
                    f_filepath, f_filename = generate_file_path_and_name(
                        self.alpha, theta, num_layer, i, self.n_qubits,
                        'fidelity', task_type, self.data_dir)
                    dump_file(fidelity, f_filepath, f_filename)
                    all_fidelity.append(fidelity)
                ave_fid = plot_func(all_fidelity)
                plt.plot(np.arange(len(ave_fid)), ave_fid)
                plt.xlabel('iterations')
                plt.ylabel('fidelity')
                fig_filepath = os.path.join(
                    self.data_dir,
                    f'data/alpha={self.alpha}/theta={theta}/plot_for_{num_layer}layer/'
                )
                fig_filename = f'fig-avefid_history_N={self.n_qubits}_alpha={self.alpha}_theta={theta}_nbSz.png'
                if not os.path.exists(fig_filepath):
                    os.makedirs(fig_filepath)

                plt.savefig(os.path.join(fig_filepath, fig_filename))
                plt.clf()
                if callback is not None:
                    callback(**kwargs)


def post_processing_for_squeezing(n_qubits, task_type, total_layer, alpha):
    pass
