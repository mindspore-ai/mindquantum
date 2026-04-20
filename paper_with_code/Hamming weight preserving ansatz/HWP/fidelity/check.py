import pickle

def print_pkl_file_content(file_path):
    """加载并打印.pkl文件的内容"""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"内容 ({file_path}):")
            print(data)
    except Exception as e:
        print(f"无法加载文件 {file_path}: {e}")

# 文件路径
best_state_path = '/mnt/nas-new/home/yange/nature_comm/pack3/fidelity/best_state.pkl'
fci_state_path = '/mnt/nas-new/home/yange/nature_comm/pack3/fidelity/H2_fci_state.pkl'

# 打印文件内容
print_pkl_file_content(best_state_path)
print_pkl_file_content(fci_state_path)
from mindquantum.simulator import fidelity


with open(best_state_path, 'rb') as file:
    best_state = pickle.load(file)
with open(fci_state_path, 'rb') as file:
    fci_state = pickle.load(file)
fidelity_value = fidelity(best_state, fci_state)
print(f"Fidelity between the optimized state and FCI state: {fidelity_value}")
