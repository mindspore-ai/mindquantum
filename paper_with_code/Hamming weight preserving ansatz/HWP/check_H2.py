import os
import pickle
import csv
import numpy as np

# 设置路径
output_csv_base_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel"
name = 'H2'
best_dist = 1
factors = [0.7, 0.9, 0.95, 1, 1.05, 1.1, 1.3, 1.6, 2, 3]
randseeds = list(range(5))  # 现在有5个种子
gate_types = ["BS", "RBS", "RCNOT"]
# gate_types = ["BS"]
# gate_types = ["RBS","RCNOT"]
runid = 'new_testhiq'
point = 'H2'

base_dir = "/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/energy/"

def load_energy(file_path):
    with open(file_path, 'rb') as f:
        energy = pickle.load(f)
    return energy

# 初始化表头
# header = ['factor', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)']
header = ['factor', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)', 'Max Energy', 'Min Energy']
# 为每种门类型进行计算
for gate_type in gate_types:
    layers = 5 if gate_type == "RCNOT" else 10
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
        standard_energy_path = os.path.join(base_dir, f'{name}_{factor}.pkl')
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