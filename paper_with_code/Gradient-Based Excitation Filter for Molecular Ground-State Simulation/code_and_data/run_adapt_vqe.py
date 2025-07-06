from adapt_vqe import *
from Filter_VQE_param import get_bound_list
from build_mole import *
import numpy as np
from tqdm import tqdm
from time import time

set_device("gpu")
tol = 1e-1

print(f"run_adapt_vqe, epsilon={tol}")

def analize(geos_list:list):
    """综合分析"""
    vqe_energies, num_params, succeses = [], [], []
    vqe_energies, num_params, succeses = [], [], []
    singles, doubles, depths = [], [], []
    
    for geo in tqdm(geos_list):
        energy, success, simpled_ansatz, free_param_names,\
         single, double, depth = run_adapt_vqe(geo, tol)
    
        vqe_energies.append(energy)
        num_params.append(len(free_param_names))
        succeses.append(success)
        singles.append(single)
        doubles.append(double)
        depths.append(depth)

    print("\nVQE:\n", vqe_energies)
    print("\n成功信息\n:", succeses)
    print("\n成功率为 (%)\n:", np.mean(succeses)*100)
    print("\n VQE 参数量为:\n", num_params)
    print("\n平均参数量为:\n", np.mean(num_params))
    print("\n单激发算符数为:\n", np.mean(singles))
    print("\n双激发算符数为:\n", np.mean(doubles))
    print("\n线路深度为:\n", np.mean(depths))


print("==============================================================================")
print("H4 分子")
bound_range = [0.1, 1.1]
bound_list = get_bound_list(bound_range)
print("键长范围：\n", bound_list)
geos_list = [get_H4_geometry(dist) for dist in bound_list]
st = time()
analize(geos_list)
print(f"总计用时:{time()-st} s")

# print("==============================================================================")
# print("H4 分子")
# bound_range = [1.1, 2.4]
# bound_list = get_bound_list(bound_range)
# print("键长范围：\n", bound_list)
# geos_list = [get_H4_geometry(dist) for dist in bound_list]
# analize(geos_list)

print("==============================================================================")
print("HF 分子")
bound_range = [0.1, 1.1]
bound_list = get_bound_list(bound_range)
print("键长范围：\n", bound_list)
geos_list = [get_HF_geometry(dist) for dist in bound_list]
st = time()
analize(geos_list)
print(f"总计用时:{time()-st} s")

print("==============================================================================")
print("H2O 分子")
bound_range = [0.1, 1.2]
bound_list = get_bound_list(bound_range)
print("键长范围：\n", bound_list)
geos_list = [get_H2O_geometry(dist) for dist in bound_list]
st = time()
analize(geos_list)
print(f"总计用时:{time()-st} s")

print("==============================================================================")
print("BeH2 分子")
bound_range = [0.1, 1.4]
bound_list = get_bound_list(bound_range)
print("键长范围：\n", bound_list)
geos_list = [get_BeH2_geometry(dist) for dist in bound_list]
st = time()
analize(geos_list)
print(f"总计用时:{time()-st} s")

print("==============================================================================")
print("NH3 分子")
# bound_range = [0.1, 1.1]
# bound_list = get_bound_list(bound_range)
bound_list = [1.0]
print("键长范围：\n", bound_list)
geos_list = [get_NH3_geometry(dist) for dist in bound_list]
st = time()
analize(geos_list)
print(f"总计用时:{time()-st} s")

# print("==============================================================================")
# print("CH4 分子")
# bound_range = [0.1, 1.2]
# bound_list = get_bound_list(bound_range)
# print("键长范围：\n", bound_list)
# geos_list = [get_H4_geometry(dist) for dist in bound_list]
# analize(geos_list)



# print("==============================================================================")
# print("CH4 分子")
# bound_range = [0.02, 0.04]
# bound_list = get_bound_list(bound_range)
# print("键长范围：\n", bound_list)
# geos_list = [get_H4_geometry(dist) for dist in bound_list]
# analize(geos_list)