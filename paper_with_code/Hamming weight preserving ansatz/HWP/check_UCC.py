# import os
# import pickle
# import csv

# # 设置路径
# # best_dist=1.42
# best_dist=1.33
# factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
# name='BeH2'
# # original_data_path = '/mnt/nas-new/home/yange/nature_comm/pack3/pks/listout/analize/LiH_RCNOT.pkl'
# # original_data_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/output/testUCC'
# # standard_result_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2'
# # output_csv_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel/F2_UCCSD.csv'
# standard_result_base_path = '/mnt/nas-new/home/yange/nature_comm/pack2/energy/BeH2'
# output_csv_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel/BeH2_UCCSD.csv'

# def load_bs_energy(file_path):
#     with open(file_path, 'rb') as f:
#         bs_energy = pickle.load(f)
#     return bs_energy

# def load_fci_energy(file_path):
#     with open(file_path, 'rb') as f:
#         fci_energy = pickle.load(f)
#     return fci_energy

# # 初始化表头
# header = ['factor', 'Energy', 'Standard Energy', 'Difference (Energy - Standard)']

# # 初始化数据列表
# data = []

# # 填充数据列表
# for factor in factors:
#     file_path = f"{original_data_base_path}/{name}_{factor}.pkl"
#     if os.path.exists(file_path):
#         bs_energy = load_bs_energy(file_path)
#         energy = bs_energy[-1][-1]
#     else:
#         energy = None  # 如果文件不存在，填入空值

#     # 加载标准能量
#     standard_energy_path = os.path.join(standard_result_base_path, f'{name}_{best_dist}_{factor}.pkl')
#     if os.path.exists(standard_energy_path):
#         standard_energy = load_fci_energy(standard_energy_path)
#         print(f"Loaded standard energy from {standard_energy_path}: {standard_energy}")
#     else:
#         standard_energy = None
#         print(f"Standard energy file does not exist: {standard_energy_path}")

#     # 计算平均值与标准能量之差
#     if energy is not None and standard_energy is not None:
#         difference = energy - standard_energy
#     else:
#         difference = None

#     row = [factor, energy, standard_energy, difference]
#     data.append(row)

# # 将数据写入CSV文件
# with open(output_csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)  # 写入表头
#     writer.writerows(data)   # 写入数据行

# print(f"CSV file saved to {output_csv_path}")

import os
import pickle
import csv
import numpy as np

# 设置路径
# best_dist = 1.33
best_dist = 1
factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
# molecules = ['F2', 'BeH2', 'LiH', 'H2']
# molecules = ['LiH', 'H2']
# molecules = ['F2']
molecules = ['BeH2']
base_paths = {
    'F2': ('/mnt/nas-new/home/yange/nature_comm/pack3/output/finalUCC', '/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2'),
    'BeH2': ('/mnt/nas-new/home/yange/nature_comm/pack3/output/finalUCC', '/mnt/nas-new/home/yange/nature_comm/pack2/energy/BeH2'),
    'LiH': ('/mnt/nas-new/home/yange/nature_comm/pack3/output/finalUCC', '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/energy'),
    'H2': ('/mnt/nas-new/home/yange/nature_comm/pack3/output/finalUCC', '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/energy')
}
output_csv_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel'

def load_energy(file_path):
    with open(file_path, 'rb') as f:
        energy = pickle.load(f)
    return energy

# 为每个分子进行计算
for name in molecules:
    original_data_base_path, standard_result_base_path = base_paths[name]
    output_csv_path = os.path.join(output_csv_base_path, f'{name}_UCCSD.csv')

    # 初始化表头
    header = ['factor', 'Mean Energy', 'Min Energy', 'Max Energy','Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)', 'Difference (Min - Standard)']

    # 初始化数据列表
    data = []

    # 填充数据列表
    for factor in factors:
        energies = []
        for seed in range(5):  
            file_path = f"{original_data_base_path}/{name}_{factor}_seed{seed}.pkl"
            if os.path.exists(file_path):
                energy_data = load_energy(file_path)
                # # 假设数据格式适合直接获取所需能量
                energy_min = np.min(energy_data[-1])
                # 假设数据格式适合直接获取所需能量
                energies.append(energy_min)
            else:
                print(f"Energy file does not exist: {file_path}")

        if energies:
            mean_energy = np.mean(energies)
            max_energy = np.max(energies)
            min_energy = np.min(energies)
            std_dev = np.std(energies)
        else:
            mean_energy = None
            max_energy = None
            min_energy = None
            std_dev = None

        # 加载标准能量
        standard_energy_path = os.path.join(standard_result_base_path, f'{name}_{best_dist}_{factor}.pkl')
        # standard_energy_path = os.path.join(standard_result_base_path, f'{name}_{factor}.pkl')
        if os.path.exists(standard_energy_path):
            standard_energy = load_energy(standard_energy_path)
            print(f"Loaded standard energy from {standard_energy_path}: {standard_energy}")
        else:
            standard_energy = None
            print(f"Standard energy file does not exist: {standard_energy_path}")

        # 计算平均值与标准能量之差
        if mean_energy is not None and standard_energy is not None:
            difference_mean = mean_energy - standard_energy
            difference_min = min_energy - standard_energy
        else:
            difference_mean = None
            difference_min = None

        row = [factor, mean_energy, min_energy,max_energy, std_dev, standard_energy, difference_mean, difference_min]
        data.append(row)

    # 将数据写入CSV文件
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入表头
        writer.writerows(data)   # 写入数据行

    print(f"CSV file saved to {output_csv_path}")
