from Filter_VQE_param import *
from build_mole import *
import numpy as np
from tqdm import tqdm

set_device("cpu")

ablation = True
# ablation = False

# uccsd = True
uccsd = False

print("run_Filter_VQE_param_ablation.py")

def run_uccsd(geos_list):
    """运行 UCCSD 算法"""
    success_list = []
    energy_list = []
    for geo in geos_list:
        hf, fci, ham, n_qubits, n_electrons = get_mole_info(geo, run_fci=True)
        ansatz = uccsd_singlet_generator(n_qubits, n_electrons, False)
        circ = get_hf_circ(n_electrons) + get_ansatz_circ(ansatz)
        energy, params_value = run_vqe(ham, circ)
        success_list.append(success_check(energy, fci))
        energy_list.append(energy)
    print("\nUCCSD 能量：\n", energy_list)
    print("\nUCCSD 成功情况：\n", success_list)
    print("\nUCCSD 成功率%：\n", np.mean(success_list)*100)
    return None


def analize(geos_list:list, tol:float=0, mag:float=5, ablation:bool=True):
    """综合分析"""
    if uccsd:
        run_uccsd(geos_list)

    hf_energies, vqe_energies, fci_energies, num_params, succeses = [], [], [], [], []
    singles, doubles, depths = [], [], []
    ablation_energies, ablation_sucesses, ablation_params, ablation_ops = [], [], [], []
    for geo in tqdm(geos_list):
        main_info, ablation_info =  study_mole(geo, tol=tol, mag=mag, ablation=ablation)
        hf_energy, vqe_energy, fci_energy, num_param, success, single, double, depth  = main_info
        ablation_energy, ablation_success, ablation_param, ablation_op = ablation_info

        hf_energies.append(hf_energy)
        vqe_energies.append(vqe_energy)
        fci_energies.append(fci_energy)
        num_params.append(num_param)
        succeses.append(success)
        singles.append(single)
        doubles.append(double)
        depths.append(depth)

        ablation_energies.append(ablation_energy)
        if ablation_success == []: # 如果没有参数可以消融，就使用 Filter-VQE 的结果
            ablation_success = [success]
        ablation_sucesses.append(ablation_success)

        ablation_params.append(ablation_param)
        ablation_ops.append(ablation_op)
    
    print(" HF:\n", hf_energies)
    print("\nVQE:\n", vqe_energies)
    print("\nFCI:\n", fci_energies)
    print("\n成功信息:\n", succeses)
    print("\n成功率为 (%):\n", np.mean(succeses)*100)
    print("\n VQE 参数量为:\n", num_params)
    print("\n平均参数量为:\n", np.mean(num_params))
    print("\n单激发算符数量为:\n", singles)
    print("\n单激发算符数平均值:\n", np.mean(singles))
    print("\n双激发算符数量为:\n", doubles)
    print("\n双激发算符数平均值:\n", np.mean(doubles))
    print("\n线路深度为:\n", np.mean(depths))
    
    if ablation:
        print("\n消融能量:\n", ablation_energies)
        print("\n消融成功信息:\n", ablation_sucesses)
        ablation_param_nums = ablation_min_count(ablation_sucesses, num_params)
        print("\n消融参数量为:\n", list(ablation_param_nums))
        print("\n平均消融参数量:\n", ablation_param_nums.mean())
        ablation_sucesses = ablation_padding(ablation_sucesses)
        print("\n消融成功率 (%)\n:", np.mean(ablation_sucesses, axis=0)*100)
        return list(ablation_param_nums)
    return None


def min_params_test(geos_list:list, params_num_list:list):
    """使用最小参数量运行VQE"""
    hf_energies, vqe_energies, fci_energies, num_params, succeses = [], [], [], [], []
    singles, doubles, depths = [], [], []
    for geo, num in tqdm(zip(geos_list, params_num_list)):
        hf_energy, vqe_energy, fci_energy, num_param, success, single, double, depth\
              = study_given_params_num(geo, num)
        
        hf_energies.append(hf_energy)
        vqe_energies.append(vqe_energy)
        fci_energies.append(fci_energy)
        num_params.append(num_param)
        succeses.append(success)
        singles.append(single)
        doubles.append(double)
        depths.append(depth)
    print("消融之后的结果：")
    print(" HF:\n", hf_energies)
    print("\nVQE:\n", vqe_energies)
    print("\nFCI:\n", fci_energies)
    print("\n成功信息:\n", succeses)
    print("\n成功率为 (%):\n", np.mean(succeses)*100)
    print("\n VQE 参数量为:\n", num_params)
    print("\n平均参数量为:\n", np.mean(num_params))
    print("\n单激发算符数量为:\n", singles)
    print("\n单激发算符数平均值:\n", np.mean(singles))
    print("\n双激发算符数量为:\n", doubles)
    print("\n双激发算符数平均值:\n", np.mean(doubles))
    print("\n线路深度为:\n", np.mean(depths))
    return None



print("==============================================================================")
print("H4 分子")
bound_range = [0.1, 1.1]
bound_list = get_bound_list(bound_range, 21)
print("键长范围:\n", bound_list)
geos_list = [get_H4_geometry(dist) for dist in bound_list]
min_param_nums = analize(geos_list)
min_params_test(geos_list, min_param_nums)


# print("==============================================================================")
# print("HF 分子")
# bound_range = [0.6, 2.4]
# bound_list = get_bound_list(bound_range, 11)
# print("键长范围:\n", bound_list)
# geos_list = [get_HF_geometry(dist) for dist in bound_list]
# min_param_nums = analize(geos_list)
# min_params_test(geos_list, min_param_nums)


# print("==============================================================================")
# print("H2O 分子")
# bound_range = [0.6, 2.4]
# bound_list = get_bound_list(bound_range, 11)
# print("键长范围:\n", bound_list)
# geos_list = [get_H2O_geometry(dist) for dist in bound_list]
# min_param_nums = analize(geos_list)
# min_params_test(geos_list, min_param_nums)



# print("==============================================================================")
# print("BeH2 分子")
# bound_range = [0.8, 2.5]
# bound_list = get_bound_list(bound_range, 11)
# print("键长范围:\n", bound_list)
# geos_list = [get_BeH2_geometry(dist) for dist in bound_list]
# min_param_nums = analize(geos_list)
# min_params_test(geos_list, min_param_nums)



# print("==============================================================================")
# print("NH3 分子")
# bound_range = [0.7, 2.4]
# bound_list = get_bound_list(bound_range, 11)
# print("键长范围:\n", bound_list)
# geos_list = [get_NH3_geometry(dist) for dist in bound_list]
# min_param_nums = analize(geos_list)
# min_params_test(geos_list, min_param_nums)
