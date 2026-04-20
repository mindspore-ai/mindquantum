import os
import pickle
import time
import numpy as np
import csv

# 定义点和对应参数
points = ['F2', 'BeH2', 'LiH', 'H2', 'H2O']
energy_list = [
    '/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2',
    '/mnt/nas-new/home/yange/nature_comm/pack2/energy/BeH2',
    '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/energy',
    '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/energy',
    '/mnt/nas-new/home/yange/nature_comm/pack3/energy/H2O'
]
Operatorpath = [
    '/mnt/nas-new/home/yange/nature_comm/pack3/data/F2',
    '/mnt/nas-new/home/yange/nature_comm/pack2/data/BeH2',
    '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/data',
    '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/data',
    '/mnt/nas-new/home/yange/nature_comm/pack3/data/H2O'
]

# 设置计算参数
point = 'H2O'
GPU = 3
pointid = -1
for id, name in enumerate(points):
    if name == point:
        pointid = id
if pointid == -1:
    raise ValueError("Invalid point name")

qubits_list = [20, 14, 12, 4, 14]
k_list = [18, 6, 4, 2, 10]
bound_dist = [1.42, 1.33, 1.60, 0.74, 0.958]

runid = 'final'
coder = 'UCC'
runid += coder
nowtype = 'r'

if not os.path.exists(f'output/{runid}'):
    os.mkdir(f'output/{runid}')

# factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
# 运行计算部分
for factor in factors:
    for seed in range(10):
        Dict = {
            # '--Hamiltonian_path': f'{Operatorpath[pointid]}/{point}_{bound_dist[pointid]}_{factor}.data',
            # '--Lowest_energy_path': f'{energy_list[pointid]}/{point}_{bound_dist[pointid]}_{factor}.pkl',
            '--Hamiltonian_path': f'{Operatorpath[pointid]}/{point}_{factor}.data',
            '--Lowest_energy_path': f'{energy_list[pointid]}/{point}_{factor}.pkl',
            '--n_qubits': qubits_list[pointid],
            '--n_electrons': k_list[pointid],
            '--Randseed': seed,
            '--Output_path': f'output/{runid}/{point}_{factor}_seed{seed}.pkl'
        }

        if nowtype == 'r':
            cmd = f'CUDA_VISIBLE_DEVICES={GPU} nohup python -u UCCSD.py '
            for key, value in Dict.items():
                cmd += f' {key} {value} '
            logpath = f'output/{runid}/{point}_{factor}_seed{seed}.log'
            if os.path.exists(logpath):
                continue
            cmd += f'> {logpath} 2>&1 &'
            print(cmd)
            os.system(cmd)
            running_list = [Dict['--Output_path']]
            while len(running_list) >= 1:
                for i in range(len(running_list)):
                    time.sleep(5)
                    if os.path.exists(running_list[i]):
                        del running_list[i]
                        break
                print("Waiting For Result")
            time.sleep(0.1)

# 加载能量文件并计算统计信息
def load_energy(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data[-1][-1]

header = ['factor', 'Average Energy', 'Min Energy', 'Max Energy', 'Std Deviation', 'Standard Energy']
data = []

for factor in factors:
    energies = []
    for seed in range(10):
        file_path = f'output/{runid}/{point}_{factor}_seed{seed}.pkl'
        if os.path.exists(file_path):
            energy = load_energy(file_path)
            energies.append(energy)
    
    if len(energies) == 0:
        continue

    average_energy = np.mean(energies)
    min_energy = np.min(energies)
    max_energy = np.max(energies)
    std_deviation = np.std(energies)

    standard_energy_path = os.path.join(energy_list[pointid], f'{point}_{bound_dist[pointid]}_{factor}.pkl')
    if os.path.exists(standard_energy_path):
        with open(standard_energy_path, 'rb') as f:
            standard_energy = pickle.load(f)
    else:
        standard_energy = None

    row = [factor, average_energy, min_energy, max_energy, std_deviation, standard_energy]
    data.append(row)

output_csv_path = f'/mnt/nas-new/home/yange/nature_comm/pack3/excel/{point}_UCCSD.csv'

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)

print(f"CSV file saved to {output_csv_path}")
