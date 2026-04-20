import os
import pickle
import csv

# 设置路径

output_csv_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel/bs_energy.csv"
name = 'H2O'
best_dist = 0.958
factors = [0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.5]
angles = [40, 60, 80, 90, 100, 104.5, 110, 120, 150, 180]

def load_bs_energy(file_path):
    with open(file_path, 'rb') as f:
        bs_energy = pickle.load(f)
    return bs_energy

# 初始化表头
header = ['Distance (Å)'] + [f'{angle}°' for angle in angles]

# 初始化数据列表
data = []
base_dir = "/mnt/nas-new/home/yange/nature_comm/pack3/output"
point = 'H2O'
runid = 'new_testhiq'
layers = 160
# 填充数据列表
for factor in factors:
    row = [factor * best_dist]  # 第一列是距离
    for angle in angles:
        file_path = os.path.join(base_dir, f'{runid}/{point}/{factor}_{angle}_BS_s2_l{layers}.pkl')
        if os.path.exists(file_path):
            bs_energy = load_bs_energy(file_path)
            row.append(bs_energy[-1][-1])
        else:
            row.append(None)  # 如果文件不存在，填入空值
    data.append(row)

# 将数据写入CSV文件
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # 写入表头
    writer.writerows(data)   # 写入数据行

print(f"CSV file saved to {output_csv_path}")
