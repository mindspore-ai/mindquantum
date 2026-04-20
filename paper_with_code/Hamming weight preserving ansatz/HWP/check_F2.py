# import os
# import pickle
# import csv
# import numpy as np

# # 设置路径
# output_csv_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel/F2_BS_mean_energy.csv"
# name = 'F2'
# best_dist = 1.42
# factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
# randseeds = list(range(10))
# Dict = {
#     '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack3/output/{}/{}/{}_{}_s{}_l{}.pkl',
#     '--Gate_type': "BS"
# }
# runid = 'new_testhiq'
# point = 'F2'
# layers = 110
# base_dir = "/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2"

# def load_bs_energy(file_path):
#     with open(file_path, 'rb') as f:
#         bs_energy = pickle.load(f)
#     return bs_energy

# def load_fci_energy(file_path):
#     with open(file_path, 'rb') as f:
#         fci_energy = pickle.load(f)
#     return fci_energy

# # 初始化表头
# header = ['factor', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)']

# # 初始化数据列表
# data = []

# # 填充数据列表
# for factor in factors:
#     energies = []
#     for seed in randseeds:
#         file_path = Dict['--Output_path'].format(runid, point, factor, Dict['--Gate_type'], seed, layers)
#         if os.path.exists(file_path):
#             bs_energy = load_bs_energy(file_path)
#             energies.append(bs_energy[-1][-1])
#         else:
#             energies.append(None)  # 如果文件不存在，填入空值
#     # 计算平均值和标准差，排除 None 值
#     valid_energies = [e for e in energies if e is not None]
#     if valid_energies:
#         mean_energy = np.mean(valid_energies)
#         std_energy = np.std(valid_energies)
#     else:
#         mean_energy = None
#         std_energy = None

#     # 加载标准能量
#     standard_energy_path = os.path.join(base_dir, f'{name}_{best_dist}_{factor}.pkl')
#     if os.path.exists(standard_energy_path):
#         standard_energy = load_fci_energy(standard_energy_path)
#         print(f"Loaded standard energy from {standard_energy_path}: {standard_energy}")
#     else:
#         standard_energy = None
#         print(f"Standard energy file does not exist: {standard_energy_path}")

#     # 计算平均值与标准能量之差
#     if mean_energy is not None and standard_energy is not None:
#         difference = mean_energy - standard_energy
#     else:
#         difference = None

#     row = [factor, mean_energy, std_energy, standard_energy, difference]
#     data.append(row)

# # 将数据写入CSV文件
# with open(output_csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)  # 写入表头
#     writer.writerows(data)   # 写入数据行

# print(f"CSV file saved to {output_csv_path}")




# import os
# import pickle
# import csv

# # 设置路径
# output_csv_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel/F2_RBS_energy.csv"
# name = 'F2'
# best_dist = 1.42
# factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
# seed = 2  # 只针对一个种子
# Dict = {
#     '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack3/output/{}/{}/{}_{}_s{}_l{}.pkl',
#     '--Gate_type': "RBS"
# }
# runid = 'new_testhiq'
# point = 'F2'
# layers = 110
# base_dir = "/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2"

# def load_bs_energy(file_path):
#     with open(file_path, 'rb') as f:
#         bs_energy = pickle.load(f)
#     return bs_energy

# def load_fci_energy(file_path):
#     with open(file_path, 'rb') as f:
#         fci_energy = pickle.load(f)
#     return fci_energy

# # 初始化表头
# # header = ['Distance (Å)', 'Energy', 'Standard Energy', 'Difference (Energy - Standard)']
# header = ['factor', 'Energy', 'Standard Energy', 'Difference (Energy - Standard)']
# # 初始化数据列表
# data = []

# # 填充数据列表
# for factor in factors:
#     file_path = Dict['--Output_path'].format(runid, point, factor, Dict['--Gate_type'], seed, layers)
#     if os.path.exists(file_path):
#         bs_energy = load_bs_energy(file_path)
#         energy = bs_energy[-1][-1]
#     else:
#         energy = None  # 如果文件不存在，填入空值

#     # 加载标准能量
#     standard_energy_path = os.path.join(base_dir, f'{name}_{best_dist}_{factor}.pkl')
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


# import os
# import pickle
# import csv
# import numpy as np

# # 设置路径
# output_csv_base_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel"
# name = 'F2'
# best_dist = 1.42
# factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
# randseeds = list(range(5))  # 现在有5个种子
# # gate_types = ["BS", "RBS", "RCNOT"]
# # gate_types = ["BS"]
# # gate_types = ["RBS"]
# gate_types = ["RCNOT"]
# runid = 'new_testhiq'
# point = 'F2'

# base_dir = "/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2"

# def load_energy(file_path):
#     with open(file_path, 'rb') as f:
#         energy = pickle.load(f)
#     return energy

# # 初始化表头
# # header = ['factor', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)']
# header = ['factor', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)', 'Max Energy', 'Min Energy']
# # 为每种门类型进行计算
# for gate_type in gate_types:
#     layers = 55 if gate_type == "RCNOT" else 110
#     output_csv_path = os.path.join(output_csv_base_path, f'{name}_{gate_type}_energy.csv')
    
#     # 初始化数据列表
#     data = []

#     # 填充数据列表
#     for factor in factors:
#         energies = []
#         for seed in randseeds:
#             file_path = f'/mnt/nas-new/home/yange/nature_comm/pack3/output/{runid}/{point}/{factor}_{gate_type}_s{seed}_l{layers}.pkl'
#             if os.path.exists(file_path):
#                 energy_data = load_energy(file_path)
#                 # 假设数据格式适合直接获取所需能量
#                 energies.append(energy_data[-1][-1])
#             else:
#                 print(f"Energy file does not exist: {file_path}")

#         # 计算平均值和标准差
#         if energies:
#             mean_energy = np.mean(energies)
#             std_energy = np.std(energies)
#             max_energy = np.max(energies)  # Calculate the maximum energy
#             min_energy = np.min(energies)
#             print(f"Calculated mean_energy: {mean_energy}")
#         else:
#             mean_energy = None
#             std_energy = None

#         # 加载标准能量
#         standard_energy_path = os.path.join(base_dir, f'{name}_{best_dist}_{factor}.pkl')
#         if os.path.exists(standard_energy_path):
#             standard_energy = load_energy(standard_energy_path)
#             print(f"Loaded standard energy from {standard_energy_path}: {standard_energy}")
#         else:
#             standard_energy = None
#             print(f"Standard energy file does not exist: {standard_energy_path}")

#         # 计算平均值与标准能量之差
#         if mean_energy is not None and standard_energy is not None:
#             difference = mean_energy - standard_energy
#         else:
#             difference = None

#         # row = [factor, mean_energy, std_energy, standard_energy, difference]
#         row = [factor, mean_energy, std_energy, standard_energy, difference, max_energy, min_energy]
#         data.append(row)

#     # 将数据写入CSV文件
#     with open(output_csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(header)  # 写入表头
#         writer.writerows(data)   # 写入数据行

#     print(f"CSV file saved to {output_csv_path}")


import os
import pickle
import csv
import numpy as np

# 设置路径
output_csv_base_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel"
name = 'F2'
best_dist = 1.42
factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
randseeds = list(range(5))  # 现在有5个种子
gate_types = ["BS", "RBS"]
# gate_types = ["BS"]
# gate_types = ["RBS","RCNOT"]
runid = 'new_testhiq'
point = 'F2'

base_dir = "/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2"

def load_energy(file_path):
    with open(file_path, 'rb') as f:
        energy = pickle.load(f)
    return energy

# 初始化表头
# header = ['factor', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)']
header = ['factor', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)', 'Max Energy', 'Min Energy']
# 为每种门类型进行计算
for gate_type in gate_types:
    layers = 100 if gate_type == "RCNOT" else 200
    output_csv_path = os.path.join(output_csv_base_path, f'{name}_{gate_type}_energy.csv')
    
    # 初始化数据列表
    data = []

    # 填充数据列表
    for factor in factors:
        energies = []
        for seed in randseeds:
            file_path = f'/mnt/nas-new/home/yange/nature_comm/pack3/output/{runid}/{point}/{factor}_{gate_type}_s{seed}_l{layers}.pkl'
            if os.path.exists(file_path):
                energy_data = load_energy(file_path)
                energy_min = np.min(energy_data[-1])
                energies.append(energy_min)
            else:
                print(f"Energy file does not exist: {file_path}")

        # 计算平均值和标准差
        if energies:
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)
            max_energy = np.max(energies)  # Calculate the maximum energy
            min_energy = np.min(energies)
            print(f"Calculated mean_energy: {mean_energy}")
        else:
            mean_energy = None
            std_energy = None

        # 加载标准能量
        standard_energy_path = os.path.join(base_dir, f'{name}_{best_dist}_{factor}.pkl')
        if os.path.exists(standard_energy_path):
            standard_energy = load_energy(standard_energy_path)
            print(f"Loaded standard energy from {standard_energy_path}: {standard_energy}")
        else:
            standard_energy = None
            print(f"Standard energy file does not exist: {standard_energy_path}")

        # 计算平均值与标准能量之差
        if mean_energy is not None and standard_energy is not None:
            difference = mean_energy - standard_energy
        else:
            difference = None

        # row = [factor, mean_energy, std_energy, standard_energy, difference]
        row = [factor, mean_energy, std_energy, standard_energy, difference, max_energy, min_energy]
        data.append(row)

    # 将数据写入CSV文件
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入表头
        writer.writerows(data)   # 写入数据行

    print(f"CSV file saved to {output_csv_path}")