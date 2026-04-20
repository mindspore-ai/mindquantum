
import pickle
original_data_path = '/mnt/nas-new/home/yange/nature_comm/pack3/pks/listout/RCNOT/LiH/LiH_3_RCNOT_seed4_hea.pkl'
with open(original_data_path, 'rb') as file:
    original_data = pickle.load(file)
print(original_data)


# import pickle

# # Path to the pickle file
# file_path = '/mnt/nas-new/home/yange/nature_comm/pack3/energy/H2O/H2O_0.958_1.3_180.pkl'

# # Load the single value from the pickle file
# with open(file_path, 'rb') as f:
#     original_value = pickle.load(f)

# print("Original Value:", original_value)

# # Set the new value
# new_value = -74.808847200696

# # Save the new value back to the pickle file
# with open(file_path, 'wb') as f:
#     pickle.dump(new_value, f)

# print(f"File {file_path} has been updated successfully.")












# import pickle
# import csv

# # original_data_path = '/mnt/nas-new/home/yange/nature_comm/pack3/pks/listout/analize/H2_RCNOT.pkl'
# original_data_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/output/testUCC'
# standard_result_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/energy/F2'
# output_csv_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel/F2_UCCSD.csv'

# with open(original_data_path, 'rb') as file:
#     original_data = pickle.load(file)


# # distances = [0.3, 0.4, 0.7, 0.9, 0.95, 1.1, 1.3, 1.05, 1.6, 1, 2, 3]
# distances = [0.7, 0.9, 0.95, 1.1, 1.3, 1.05, 1.6, 1, 2, 3]
# best_dist=1.42

# with open(output_csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入标题行
#     writer.writerow(['Distance', 'Original Result', 'Delta'])
    
#     # 循环加载标准结果并计算差距
#     for i, dist in enumerate(distances):
#         # 构建每个标准结果的文件路径
#         standard_result_path = f"{standard_result_base_path}/F2_{best_dist}_{dist}.pkl"
#         with open(standard_result_path, 'rb') as file:
#             standard_result = pickle.load(file)

#         original_data_path = f"{original_data_base_path}/F2_{dist}.pkl"
#         with open(original_data_path, 'rb') as file:
#             original_result = pickle.load(file)
#         # 计算差距
#         # original_result = original_data[i]['heaRCNOT']
#         delta = original_result - standard_result
        
#         # 写入每行数据
#         writer.writerow([dist, original_result, delta, standard_result])

# print(f"Results have been saved to {output_csv_path}")

# import pickle
# import csv

# # 路径设置
# # original_data_path = '/mnt/nas-new/home/yange/nature_comm/pack3/pks/listout/analize/LiH_BS.pkl'
# # standard_result_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/energy'
# # output_csv_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel/LiH_BS.csv'
# original_data_path = '/mnt/nas-new/home/yange/nature_comm/pack3/pks/listout/analize/H2_RCNOT.pkl'
# standard_result_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/energy'
# output_csv_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel/H2_rcnot.csv'
# # 加载原始数据
# with open(original_data_path, 'rb') as file:
#     original_data = pickle.load(file)

# # 键长列表
# distances = [0.3, 0.4, 0.7, 0.9, 0.95, 1.1, 1.3, 1.05, 1.6, 1, 2, 3]
# methods = ['FCRBS', 'FCBS', 'newheaRBS', 'newheaBS']
# # methods = ['testnewheaRBS', 'testnewheaBS', 'FCRBS', 'FCBS']

# # 准备 CSV 文件写入
# with open(output_csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入标题行
#     writer.writerow(['Distance', 'Method', 'Original Result', 'Standard Result', 'Delta'])

#     # 循环加载标准结果并计算差距
#     for i, dist in enumerate(distances):
#         # 构建每个标准结果的文件路径
#         standard_result_path = f"{standard_result_base_path}/H2_{dist}.pkl"
#         with open(standard_result_path, 'rb') as file:
#             standard_result = pickle.load(file)
#         print(standard_result)
#         # 对每个方法处理结果
#         for method in methods:
#             original_result = original_data[i][method]
#             delta = original_result - standard_result
            
#             # 写入每行数据
#             writer.writerow([dist, method, original_result, standard_result, delta])

# print(f"Results have been saved to {output_csv_path}")

# import pickle
# import csv

# # 路径设置
# # original_data_path = '/mnt/nas-new/home/yange/nature_comm/pack3/pks/listout/analize/LiH_BS.pkl'
# # standard_result_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/LiH/energy'
# # output_csv_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel/LiH_BS.csv'
# original_data_path = '/mnt/nas-new/home/yange/nature_comm/pack3/pks/listout/analize/H2_RCNOT.pkl'
# standard_result_base_path = '/mnt/nas-new/home/yange/nature_comm/pack3/data/pks_data/H2/energy'
# output_csv_path = '/mnt/nas-new/home/yange/nature_comm/pack3/excel/H2_rcnot.csv'
# # 加载原始数据
# with open(original_data_path, 'rb') as file:
#     original_data = pickle.load(file)

# # 键长列表
# distances = [0.3, 0.4, 0.7, 0.9, 0.95, 1.1, 1.3, 1.05, 1.6, 1, 2, 3]
# methods = ['FCRBS', 'FCBS', 'newheaRBS', 'newheaBS']


# # 准备 CSV 文件写入
# with open(output_csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入标题行
#     writer.writerow(['Distance', 'Original Result', 'Standard Result', 'Delta'])

#     # 循环加载标准结果并计算差距
#     for i, dist in enumerate(distances):
#         # 构建每个标准结果的文件路径
#         standard_result_path = f"{standard_result_base_path}/H2_{dist}.pkl"
#         with open(standard_result_path, 'rb') as file:
#             standard_result = pickle.load(file)
#         # print(standard_result)

#         original_result = original_data[i]['heaRCNOT']
#         delta = original_result - standard_result
            
#         writer.writerow([dist,original_result, standard_result, delta])

# print(f"Results have been saved to {output_csv_path}")