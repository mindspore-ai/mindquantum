"""公共函数"""
import numpy as np
from mindquantum import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt

Device = "mqvector"
Debug = False # 是否开启 debug 模式. 若为 True 则会打印中间信息
chem_acc = 0.0016 # 化学精度


def count_ops(ops:list):
    """统计拟设中单激发算符和双激发算符数量"""
    single_count, double_count = 0, 0
    for op in ops:
        if len(op) == 2:
            single_count += 1
        else:
            double_count += 1
    return single_count, double_count

def get_bound_list(bound_range:list, num_points:int=21):
    """获取键长列表"""
    return [bound_range[0] + i*(bound_range[1]-bound_range[0])/(num_points-1) for i in range(num_points)]


def set_device(device:str="cpu"):
    """可选 cpu, 或者 gpu"""
    global Device
    if device == "gpu":
        Device = "mqvector_gpu"
    else:
        Device = "mqvector"
    return None


def fun(p0, grad_ops):
    """优化函数"""
    f, g = grad_ops(p0)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]            
    return f, g
    

def run_vqe(ham:Hamiltonian, 
            circ:Circuit):
    """根据给定的量子线路运行 VQE 算法, 求基态能级. """
    p0 = [0.] * len(circ.params_name) # 参数全 0 初始化
    grad_ops = Simulator(Device, ham.n_qubits).get_expectation_with_grad(ham, circ)
    res = minimize(fun, p0, args=(grad_ops), method='bfgs', jac=True)
    return res.fun, res.x


def ablation_padding(lsts:list):
    """对消融实验的成功信息进行自动补全
    >>> lsts = [[True], [True, False], [False, False, False]]
    >>> ablation_padding(lsts)
    [[True, True, True], [True, False, False], [False, False, False]]
    """
    length = max([len(term) for term in lsts])
    new_lsts = [term + [term[-1]] * (length-len(term)) for term in lsts]
    return new_lsts



def take_uccsd_test(geometry:list):
    """测试 UCCSD 拟设的效果. """
    _, fci_energy, ham, n_qubits, n_electrons = get_mole_info(geometry, run_fci=True)

    hf_circ = get_hf_circ(n_electrons)
    ansatz = uccsd_singlet_generator(n_qubits, n_electrons, anti_hermitian=False)
    circ = hf_circ + get_ansatz_circ(ansatz)
    uccsd_energy, _ = run_vqe(ham, circ)

    return uccsd_energy, success_check(uccsd_energy, fci_energy)


def ablation_min_count(sucesses, num_list:list):
    """根据消融实验结果来确定参数数量或算符数量的极限值"""
    cut_nums = []
    for sub in sucesses:
        cut_nums.append(max([0] + [i+1 for i, x in enumerate(sub) if x]))
    return np.array(num_list) - np.array(cut_nums)


def get_equilibrium(dist_list, func):
    """计算平衡键长，预览势能曲线"""
    from Filter_VQE_param import get_mole_info
    fci_list = []
    hf_list = []
    for dist in dist_list:
        geo = func(dist)
        hf, fci, _, _, _ = get_mole_info(geo, run_fci=True)
        fci_list.append(fci)
        hf_list.append(hf)
    print("平衡键长为：",dist_list[np.argmin(fci_list)])

    plt.plot(dist_list, fci_list)
    plt.plot(dist_list, hf_list)
    plt.show()
    return fci_list, hf_list