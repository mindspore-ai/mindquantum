

import os
import pickle
import csv
import numpy as np

# Constants
randseeds = list(range(5))
Dict = {
    '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack2/output/{}/{}_{}layers_{}_seed{}_{}.pkl'
    # '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack3/output/{}/{}/{}_{}_s{}_l{}.pkl'
}

# Parameters
runid = 'paramhiq'
# runid = 'layerhiq'
# points = ['F2', 'BeH2'] 
# points = ['F2'] 
points = ['BeH2'] 
# layers = [50,55,60,70,80,90,100,110,120]
# layers = [25, 30, 35, 40, 45, 50, 55, 60]
# layers = [300, 350, 400, 450, 500]
layers = [300, 350, 375, 400, 425, 450, 475, 500]
# layers = [150, 175, 200, 225, 250]
# gate_types = ['BS', 'RBS', 'RCNOT']
# gate_types = ['BS', 'RBS']
gate_types = ['BS']
# gate_types = ['RCNOT']
circuit_type = 'newhea'

# Functions to load data
def load_energy(file_path):
    with open(file_path, 'rb') as f:
        energy = pickle.load(f)
    return energy

def load_standard_energy(file_path):
    with open(file_path, 'rb') as f:
        fci_energy = pickle.load(f)
    return fci_energy

# Function to process data for a given gate type
def process_layer_data(point, gate_type, circuit_type, layers, runid, randseeds):
    # Define output path for each molecule and gate type
    output_csv_path = f"/mnt/nas-new/home/yange/nature_comm/pack3/excel/{point}_{gate_type}_layer.csv"

    # Initialize the header
    header = ['Layers', 'Gate Type', 'Mean Energy', 'Standard Deviation', 'Max Energy', 'Min Energy', 'Standard Energy', 'Difference (Mean - Standard)','Difference (Min - Standard)', 'Log Mean Difference', 'Log Standard Deviation']
    data = []
    # Load standard energy
    standard_energy_path = os.path.join(f"/mnt/nas-new/home/yange/nature_comm/pack2/energy/{point}_dist1.34.pkl")
    # standard_energy_path = os.path.join(f"/mnt/nas-new/home/yange/nature_comm/pack3/energy/{point}_dist1.42.pkl")
    print(f"Loading standard energy from: {standard_energy_path}")
    if os.path.exists(standard_energy_path):
        standard_energy = load_standard_energy(standard_energy_path)
        print(f"Loaded standard energy: {standard_energy}")
    else:
        standard_energy = None
        print(f"Standard energy file does not exist: {standard_energy_path}")
    # Fill data list
    for layer in layers:
        energies = []
        log_differences = []
        for seed in randseeds:
            # file_path = Dict['--Output_path'].format(runid, point, gate_type, circuit_type, seed, layer)
            file_path = Dict['--Output_path'].format(runid, point, layer, gate_type, seed, circuit_type)
            print(f"Trying to access file: {file_path}") 
            if os.path.exists(file_path):
                print(f"File exists: {file_path}") 
                energy = load_energy(file_path)
                print(f"Successfully loaded energy data from: {file_path}")
                # final_energy = energy[-1][-1]
                final_energy = np.min(energy[-1])
                energies.append(final_energy)
                if standard_energy is not None:
                    difference = final_energy - standard_energy
                    log_differences.append(np.log10(np.abs(difference)))
            else:
                print(f"File does not exist: {file_path}") 
                energies.append(None)
                log_differences.append(None)

        # Calculate mean, std, max, and min excluding None values
        valid_energies = [e for e in energies if e is not None]
        print(f"Energies for layer {layer}: {energies}")
        valid_log_differences = [d for d in log_differences if d is not None]
        if valid_energies:
            mean_energy = np.mean(valid_energies)
            std_energy = np.std(valid_energies)
            max_energy = np.max(valid_energies)
            min_energy = np.min(valid_energies)
            min_difference=min_energy-standard_energy
        else:
            mean_energy = None
            std_energy = None
            max_energy = None
            min_energy = None
            min_difference=None


        if valid_log_differences:
            log_mean_difference = np.mean(valid_log_differences)
            log_std_difference = np.std(valid_log_differences)
        else:
            log_mean_difference = None
            log_std_difference = None



        # Calculate difference
        if mean_energy is not None and standard_energy is not None:
            difference = mean_energy - standard_energy
        else:
            difference = None

        row = [layer, gate_type, mean_energy, std_energy, max_energy, min_energy, standard_energy, difference, min_difference, log_mean_difference, log_std_difference]
        data.append(row)

    # Write data to CSV file
    print(f"Data to write for {output_csv_path}: {data}") 
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header
        writer.writerows(data)   # Write data rows

    print(f"CSV file saved to {output_csv_path}")

# Process each molecule and gate type
for point in points:
    for gate_type in gate_types:
        process_layer_data(point, gate_type, circuit_type, layers, runid, randseeds)



# import os
# import pickle
# import csv
# import numpy as np

# # Constants
# randseeds = list(range(5))
# Dict = {
#     # '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack2/output/{}/{}_{}layers_{}_seed{}_{}.pkl'
#     '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack3/output/{}/{}/{}_{}_s{}_l{}.pkl'
# }

# # Parameters
# runid = 'paramhiq'
# runid = 'layerhiq'
# # points = ['F2', 'BeH2'] 
# points = ['F2'] 
# # points = ['BeH2'] 
# layers = [50,55,60,70,80,90,100,110,120]
# # layers = [25, 30, 35, 40, 45, 50, 55, 60]
# # layers = [300, 350, 400, 450, 500]
# # gate_types = ['BS', 'RBS', 'RCNOT']
# gate_types = ['BS', 'RBS']
# # gate_types = ['RCNOT']
# circuit_type = 'newhea'

# # Functions to load data
# def load_energy(file_path):
#     with open(file_path, 'rb') as f:
#         energy = pickle.load(f)
#     return energy

# def load_standard_energy(file_path):
#     with open(file_path, 'rb') as f:
#         fci_energy = pickle.load(f)
#     return fci_energy

# # Function to process data for a given gate type
# def process_layer_data(point, gate_type, circuit_type, layers, runid, randseeds):
#     # Define output path for each molecule and gate type
#     output_csv_path = f"/mnt/nas-new/home/yange/nature_comm/pack3/excel/{point}_{gate_type}_layer.csv"

#     # Initialize the header
#     header = ['Layers', 'Gate Type', 'Mean Energy', 'Standard Deviation', 'Max Energy', 'Min Energy', 'Standard Energy', 'Difference (Mean - Standard)', 'Log Mean Difference', 'Log Standard Deviation']
#     data = []
#     # Load standard energy
#     # standard_energy_path = os.path.join(f"/mnt/nas-new/home/yange/nature_comm/pack2/energy/{point}_dist1.34.pkl")
#     standard_energy_path = os.path.join(f"/mnt/nas-new/home/yange/nature_comm/pack3/energy/{point}_dist1.42.pkl")
#     if os.path.exists(standard_energy_path):
#         standard_energy = load_standard_energy(standard_energy_path)
#     else:
#         standard_energy = None
#     # Fill data list
#     for layer in layers:
#         energies = []
#         log_differences = []
#         for seed in randseeds:
#             file_path = Dict['--Output_path'].format(runid, point, gate_type, circuit_type, seed, layer)
#             # file_path = Dict['--Output_path'].format(runid, point, layer, gate_type, seed, circuit_type)
#             if os.path.exists(file_path):
#                 energy = load_energy(file_path)
#                 # final_energy = energy[-1][-1]
#                 final_energy = np.min(energy[-1])
#                 energies.append(final_energy)
#                 if standard_energy is not None:
#                     difference = final_energy - standard_energy
#                     log_differences.append(np.log10(np.abs(difference)))
#             else:
#                 energies.append(None)
#                 log_differences.append(None)

#         # Calculate mean, std, max, and min excluding None values
#         valid_energies = [e for e in energies if e is not None]
#         valid_log_differences = [d for d in log_differences if d is not None]
#         if valid_energies:
#             mean_energy = np.mean(valid_energies)
#             std_energy = np.std(valid_energies)
#             max_energy = np.max(valid_energies)
#             min_energy = np.min(valid_energies)
#         else:
#             mean_energy = None
#             std_energy = None
#             max_energy = None
#             min_energy = None


#         if valid_log_differences:
#             log_mean_difference = np.mean(valid_log_differences)
#             log_std_difference = np.std(valid_log_differences)
#         else:
#             log_mean_difference = None
#             log_std_difference = None



#         # Calculate difference
#         if mean_energy is not None and standard_energy is not None:
#             difference = mean_energy - standard_energy
#         else:
#             difference = None

#         row = [layer, gate_type, mean_energy, std_energy, max_energy, min_energy, standard_energy, difference, log_mean_difference, log_std_difference]
#         data.append(row)

#     # Write data to CSV file
#     with open(output_csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(header)  # Write header
#         writer.writerows(data)   # Write data rows

#     print(f"CSV file saved to {output_csv_path}")

# # Process each molecule and gate type
# for point in points:
#     for gate_type in gate_types:
#         process_layer_data(point, gate_type, circuit_type, layers, runid, randseeds)


# import os
# import pickle
# import csv
# import numpy as np

# # 设置路径
# output_csv_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel/F2_BS_layer.csv"
# # output_csv_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel/BeH2_BS_layer.csv"
# name = 'BeH2'
# randseeds = list(range(10))
# Dict = {
#     # '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack3/output/{}/{}/{}_{}_s{}_l{}.pkl',
#     '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack2/output/{}/{}_{}layers_{}_seed{}_{}.pkl',
#     '--Gate_type': "BS",
#     '--Circuit_type':'newhea'
# }
# # runid = 'layerhiq'
# # point = 'F2'
# runid = 'paramhiq'
# point = 'BeH2'
# # layers = [50,55,60,70,80,90,100,110,120]
# layers = [300,350,400,450,500]

# def load_bs_energy(file_path):
#     with open(file_path, 'rb') as f:
#         bs_energy = pickle.load(f)
#     return bs_energy

# def load_fci_energy(file_path):
#     with open(file_path, 'rb') as f:
#         fci_energy = pickle.load(f)
#     return fci_energy

# # 初始化表头
# header = ['Layers', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)']

# # 初始化数据列表
# data = []

# # 填充数据列表
# for layer in layers:
#     energies = []
#     for seed in randseeds:
#         # file_path = Dict['--Output_path'].format(runid,point, Dict['--Gate_type'],Dict['--Circuit_type'],seed,layer)
#         file_path = Dict['--Output_path'].format(runid, point, layer, Dict['--Gate_type'], seed, Dict['--Circuit_type'])
#         if os.path.exists(file_path):
#             bs_energy = load_bs_energy(file_path)
#             energies.append(bs_energy[-1][-1])
#             # print(bs_energy[-1][-1])
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
#     # standard_energy_path = os.path.join("/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2_dist1.42.pkl")
#     standard_energy_path = os.path.join("/mnt/nas-new/home/yange/nature_comm/pack2/energy/BeH2_dist1.34.pkl")
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

#     row = [layer, mean_energy, std_energy, standard_energy, difference]
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
# # output_csv_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel/F2_RCNOT_layer.csv"
# output_csv_path = "/mnt/nas-new/home/yange/nature_comm/pack3/excel/BeH2_RBS_layer.csv"
# name = 'BeH2'
# randseeds = [0]
# Dict = {
#     # '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack3/output/{}/{}/{}_{}_s{}_l{}.pkl',
#     '--Output_path': '/mnt/nas-new/home/yange/nature_comm/pack2/output/{}/{}_{}layers_{}_seed{}_{}.pkl',
#     '--Gate_type': "RBS",
#     '--Circuit_type': 'newhea'
# }
# # runid = 'layerhiq'
# runid = 'paramhiq'
# point = 'BeH2'
# # layers = [50,55,60,70,80,90,100,110,120]
# # layers = [25, 30, 35, 40, 45, 50, 55, 60]

# layers = [300,350,400,450,500]
# # layers = [150, 175, 200, 225, 250]

# def load_bs_energy(file_path):
#     with open(file_path, 'rb') as f:
#         bs_energy = pickle.load(f)
#     return bs_energy

# def load_fci_energy(file_path):
#     with open(file_path, 'rb') as f:
#         fci_energy = pickle.load(f)
#     return fci_energy

# # 初始化表头
# header = ['Layers', 'Mean Energy', 'Standard Deviation', 'Standard Energy', 'Difference (Mean - Standard)']

# # 初始化数据列表
# data = []

# # 填充数据列表
# for layer in layers:
#     energies = []
#     for seed in randseeds:
#         # file_path = Dict['--Output_path'].format(runid,point, Dict['--Gate_type'],Dict['--Circuit_type'],seed,layer)
#         file_path = Dict['--Output_path'].format(runid, point, layer, Dict['--Gate_type'], seed, Dict['--Circuit_type'])
#         print(f"Checking file: {file_path}")
#         if os.path.exists(file_path):
#             bs_energy = load_bs_energy(file_path)
#             energies.append(bs_energy[-1][-1])
#             print(f"Loaded energy from {file_path}: {bs_energy[-1][-1]}")
#         else:
#             energies.append(None)  # 如果文件不存在，填入空值
#             print(f"File does not exist: {file_path}")

#     # 计算平均值和标准差，排除 None 值
#     valid_energies = [e for e in energies if e is not None]
#     if valid_energies:
#         mean_energy = np.mean(valid_energies)
#         std_energy = np.std(valid_energies)
#     else:
#         mean_energy = None
#         std_energy = None

#     # 加载标准能量
#     standard_energy_path = os.path.join("/mnt/nas-new/home/yange/nature_comm/pack2/energy/BeH2_dist1.34.pkl")
#     # standard_energy_path = os.path.join("/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2_dist1.42.pkl")
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

#     row = [layer, mean_energy, std_energy, standard_energy, difference]
#     data.append(row)

# # 将数据写入CSV文件
# with open(output_csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)  # 写入表头
#     writer.writerows(data)   # 写入数据行

# print(f"CSV file saved to {output_csv_path}")
