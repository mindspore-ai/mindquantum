# -*- coding: utf-8 -*-
"""
使用 MindQuantum 通过高级 QAOA 变体解决 Max-Cut 问题。
该脚本整合了标准QAOA、温启动QAOA和自适应偏置QAOA (ab-QAOA)，
其中 ab-WS-QAOA 使用了论文中定义的特定Adam优化器来更新偏置场h。
由于ab-WS-QAOA在比特数n~1000时才能展现较大优势，本代码只是一个6qubit的demo
本脚本的代码可以用于检验：ab-WS-QAOA在运行伊始就有GW-warm start的切割边数
"""

import numpy as np
import networkx as nx
from scipy.optimize import minimize

# MindQuantum 核心库导入
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, RX, RY, Rzz
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator

# 从之前的脚本中导入经典求解器
from SDP import maxcut_interior_point, GW_algorithm

# ==============================================================================
# --- 1. 函数构建 ---
# ==============================================================================
def create_graph(n=6, seed=400):
    return nx.random_regular_graph(3, n, seed=seed)

def build_problem_hamiltonian(graph: nx.Graph) -> Hamiltonian:
    ham = QubitOperator()
    for i, j in graph.edges:
        ham += QubitOperator(f'Z{i} Z{j}')
    return Hamiltonian(ham)

def build_warm_start_circuit(delta: np.ndarray) -> Circuit:
    """根据 delta = alpha - pi/2 构建温启动线路。"""
    circ = Circuit([RY(delta_i).on(i) for i, delta_i in enumerate(delta)])
    # 论文中的初态是 Ry(alpha - pi/2)|0> = Ry(delta)|0>
    print(" -> 温启动线路已构建。")
    return circ

def build_adaptive_ansatz(graph: nx.Graph, p: int, alpha: np.ndarray) -> Circuit:
    """
    使用 Ry-Rx-Ry 结构构建自适应偏置混合层。

    Args:
        alpha (np.ndarray): 形状为 (p, n) 的角度数组，用于确定混合器的旋转轴。
    """
    circ = Circuit()
    for i in range(p):
        # Cost Layer
        circ += Circuit([Rzz(2*f'gamma_{i}').on([u, v]) for u, v in graph.edges])
        circ.barrier()
        # Mixer Layer: U = Ry(-alpha) * Rx(beta) * Ry(alpha)
        # 这实现了 e^(-i * beta/2 * (cos(alpha)X - sin(alpha)Z))
        for j in graph.nodes:
            circ += RY(alpha[i, j]).on(j)
            circ += RX(f'beta_{i}').on(j)
            circ += RY(-alpha[i, j]).on(j)
        circ.barrier()
    print(" -> 自适应偏置Ansatz线路已构建。")
    return circ

# ==============================================================================
# --- 2. 主逻辑与优化流程 ---
# ==============================================================================
def main_adaptive_custom(p_layers, max_iterations=30):
    """
    用于运行 Adaptive QAOA 的主函数，采用指定的交替优化策略。
    """
    print(f"\n===== 正在运行QAOA变体: ADAPTIVE (CUSTOM UPDATE) (p={p_layers}) =====")
    # --- 初始化 ---
    graph = create_graph(n=6)
    num_qubits = len(graph.nodes)
    cost_hamiltonian = build_problem_hamiltonian(graph)
    sim = Simulator('mqvector', num_qubits)
    
    # 定义Adam超参数
    _ADAM_HYPERPARAMS = {'beta1': 0.9, 'beta2': 0.999, 'ell': 0.4, 'epsilon': 1e-8}

    # --- 温启动 ---
    print(" -> 步骤1: 运行经典GW算法获取温启动解...")
    pro_info = np.array([[u, v, 1.0] for u, v in graph.edges()])
    _, _, X_opt = maxcut_interior_point(pro_info)
    v_ran = np.random.uniform(-1, 1, (1, num_qubits))
    gw_energy, gw_solution, _ = GW_algorithm(pro_info, X_opt[-1], v_ran)
    print(f" -> GW算法找到的切割边数: {gw_energy:.4f}")
    
    # --- 初始化循环中使用的状态变量 ---
    # h (偏置场)
    initial_bias = gw_solution / np.sqrt(3)
    # print(gw_solution)
    current_h = np.tile(initial_bias, p_layers) # 假设每层的h都相同
    # h的Adam动量
    m_h = np.zeros_like(current_h)
    v_h = np.zeros_like(current_h)
    # gamma和beta角度
    current_angles = [2*-1* np.pi,2*-0.5* np.pi] 

    # --- 交替优化循环 ---
    print("\n--- 开始交替优化 ---")
    for t in range(1, max_iterations + 1):
        # --- 步骤A: 固定h，优化γ和β ---
        # 1. 根据当前的h计算alpha
        current_alpha = np.arctan(current_h).reshape((p_layers, num_qubits))
        # 2. 构建温启动线路 (初态也依赖于h)
        delta = np.arctan(current_h[:num_qubits]) - np.pi / 2
        initial_circuit = build_warm_start_circuit(delta)
        # 3. 构建Ansatz (alpha作为固定值传入)
        ansatz_circuit = build_adaptive_ansatz(graph, p_layers, current_alpha)
        full_circuit = initial_circuit + ansatz_circuit
        # 4. 使用MindQuantum优化角度
        grad_ops_angles = sim.get_expectation_with_grad(cost_hamiltonian, full_circuit)
        
        def angle_objective(params):
            exp_val, grad = grad_ops_angles(params)
            return np.real(exp_val)[0, 0], np.real(grad)[0, 0]
        
        current_cost,_ = angle_objective(current_angles)
        cut_value = (len(graph.edges) - current_cost) / 2
        print(f"Iter: {t:<3} | Cost: {current_cost:>7.4f} | Approx. Cut: {cut_value:>7.4f}")
        opt_res = minimize(fun=angle_objective, x0=current_angles, method='bfgs', jac=True, options={'maxiter': 1})
        current_angles = opt_res.x
        current_cost = opt_res.fun

        # --- 步骤B: 固定γ和β，使用Adam逻辑更新h ---
        
        # 计算<Z_j>期望值
        z_expectations = []
        for j in range(num_qubits):
            z_ham = Hamiltonian(QubitOperator(f'Z{j}'))
            z_exp, _ = sim.get_expectation_with_grad(z_ham, full_circuit)(current_angles)
            z_expectations.append(np.real(z_exp))
        z_expectations = np.array(z_expectations)
        
        # 3. 根据论文公式(39)计算h的“梯度”
        # 假设所有层的h都用同样的梯度更新
        h_grad_base = current_h[:num_qubits] + z_expectations.flatten() / np.sqrt(3)
        h_grad = np.tile(h_grad_base, p_layers)
        
        # 4. 学习率调度和Adam更新
        bias_corrected_learning_rate = _ADAM_HYPERPARAMS['ell'] * np.sqrt(1 - _ADAM_HYPERPARAMS['beta2']**t) / (1 - _ADAM_HYPERPARAMS['beta1']**t)
        
        m_h = _ADAM_HYPERPARAMS['beta1'] * m_h + (1 - _ADAM_HYPERPARAMS['beta1']) * h_grad
        v_h = _ADAM_HYPERPARAMS['beta2'] * v_h + (1 - _ADAM_HYPERPARAMS['beta2']) * (h_grad ** 2)
        
        update = bias_corrected_learning_rate * m_h / (np.sqrt(v_h) + _ADAM_HYPERPARAMS['epsilon'])
        current_h -= update
    print("\n--- 优化完成 ---")
    final_cut = (len(graph.edges) - current_cost) / 2
    print(f"最终最小期望值 (Cost): {current_cost:.6f}")
    print(f"最终计算出的最大切割边数 (Max-Cut): {final_cut:.6f}")

# ==============================================================================
# ------
# ==============================================================================
if __name__ == "__main__":
    main_adaptive_custom(p_layers=1)