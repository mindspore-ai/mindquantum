import multiprocessing
import numpy as np
from train import TrainModel


def task(n_qubits, alpha, task_type, thetas, runs, total_layer):
    num_layers = range(1, total_layer)
    model = TrainModel(n_qubits, alpha)
    model.train(task_type, thetas, num_layers, runs)


# # full task
# task_vs_layer = {
#     'nbSz': 7,
#     'nnbSz': 11,
#     'nnbsoSz': 7,
#     'nnnbSz': 11,
#     'nnnbsoSz': 8,
# }
# theta_range = np.arange(0.1, np.pi / 2, 0.2)
# runs_range = np.array(range(10))
# split_theta = 10
# split_runs = 10
# n_process = split_theta * split_runs * len(task_vs_layer)

# simple test task
task_vs_layer = {
    'nbSz': 7,
}
theta_range = np.arange(0.1, np.pi / 2, 0.2)
runs_range = np.array(range(10))
split_theta = 8
split_runs = 2
n_process = split_theta * split_runs * len(task_vs_layer)

# run task
pool = multiprocessing.Pool(processes=n_process)
result = {}
for tidx, thetas in enumerate(np.array_split(theta_range, split_theta)):
    for ridx, runs in enumerate(np.array_split(runs_range, split_runs)):
        for task_type, num_layer in task_vs_layer.items():
            this_task = pool.apply_async(
                task, (10, 0, task_type, thetas, runs, num_layer))
            result[(tidx, ridx, task_type)] = this_task
pool.close()
pool.join()
for res in result.values():
    res.get()

