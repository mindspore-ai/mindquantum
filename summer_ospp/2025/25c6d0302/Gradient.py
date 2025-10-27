import numpy as np
from WSQAOA import QAOA
from typing import Dict, Any

# ----------------------------------------------------------------------------
# 1. 梯度计算函数
# ----------------------------------------------------------------------------

def gradient_function(para: np.ndarray, qaoa: QAOA, epsilon: float) -> (np.ndarray, float):
    """
    通过有限差分法计算目标函数（能量期望值）的梯度。

    这个方法通过在每个参数维度上引入一个微小的扰动 epsilon，来数值近似梯度。

    Args:
        para (np.ndarray): 当前的参数向量（例如 QAOA 的角度）。
        qaoa (QAOA): QAOA 类的实例，需要提供 get_expectation 方法。
        epsilon (float): 用于计算有限差分的微小扰动值。

    Returns:
        Tuple[np.ndarray, float]: 一个元组，包含计算出的梯度向量和在原始参数点 `para` 处的能量值。
    """
    # 首先计算在原始参数点上的能量（期望值）
    base_energy = qaoa.get_expectation(para)
    
    # 初始化一个与参数维度相同的零向量，用于存储梯度
    gradient = np.zeros_like(para)
    
    # 遍历每一个参数，计算偏导数
    for j in range(len(para)):
        # 保存当前参数的原始值，以便后续恢复
        original_param_value = para[j]
        
        # 在第 j 个参数上增加一个微小的扰动
        para[j] += epsilon
        
        # 计算扰动后的能量
        perturbed_energy = qaoa.get_expectation(para)
        
        # 恢复第 j 个参数的原始值，避免影响其他维度的计算
        para[j] = original_param_value
        
        # 根据有限差分公式计算偏导数
        # f'(x) ≈ (f(x+ε) - f(x)) / ε
        gradient[j] = (perturbed_energy - base_energy) / epsilon
    
    return gradient, base_energy

# ----------------------------------------------------------------------------
# 2. 输入输出数据结构类
# ----------------------------------------------------------------------------

class GradientInput:
    """
    一个数据类，用于封装 Adam 优化器的输入参数。
    """
    def __init__(self, para: np.ndarray, n: int, qaoa_variant: Dict, pro_solve: Any, sys_para: Dict):
        """
        初始化输入对象。

        Args:
            para (np.ndarray): 初始的 QAOA 参数。
            n (int): 系统的量子比特数。
            qaoa_variant (Dict): QAOA 变体的配置字典。
            pro_solve (Any): 问题求解器的相关配置或实例。
            sys_para (Dict): 系统参数，例如 `{'max_iters': 1000}`。
        """
        self.para = para
        self.n = n
        self.qaoa_variant = qaoa_variant
        self.pro_solve = pro_solve
        self.max_iters = sys_para['max_iters']

class GradientOutput:
    """
    一个数据类，用于封装 Adam 优化器的输出结果。
    """
    def __init__(self, para_opt: np.ndarray, energy: float, itera: int, bias_energy: float, bias_opt: np.ndarray, others: Dict):
        """
        初始化输出对象。

        Args:
            para_opt (np.ndarray): 找到的最优参数。
            energy (float): 对应的最优能量。
            itera (int): 实际执行的迭代次数。
            bias_energy (float): 最终的偏置能量。
            bias_opt (np.ndarray): 最终的偏置项。
            others (Dict): 包含详细历史记录的字典（如参数、能量、偏置等）。
        """
        self.para_opt = para_opt
        self.energy = energy
        self.itera = itera
        self.bias_energy = bias_energy
        self.bias_opt = bias_opt
        self.others = others

# ----------------------------------------------------------------------------
# 3. Adam 优化器主函数
# ----------------------------------------------------------------------------

def adam_iters(gradient: GradientInput) -> GradientOutput:
    """
    使用 Adam 优化算法寻找 QAOA 的最优参数。

    该函数会执行迭代优化，直到满足收敛条件或达到最大迭代次数。

    Args:
        gradient (GradientInput): 包含所有优化所需配置的输入对象。

    Returns:
        GradientOutput: 包含优化结果的输出对象。
    """
    # --- 1. 定义超参数和配置 ---
    # 将所有超参数集中定义，便于管理和调试
    _ADAM_HYPERPARAMS = {
        'learning_rate': 0.02,   # 学习率 alpha
        'beta1': 0.9,            # 一阶矩估计的指数衰减率
        'beta2': 0.999,          # 二阶矩估计的指数衰减率
        'epsilon': 1e-8,         # 防止除以零的小常数
        'gradient_precision': 1e-5 # 用于 gradient_function 的有限差分精度
    }
    
    _CONVERGENCE_THRESHOLDS = {
        'energy': 1e-3,          # 能量变化收敛阈值
        'param_change': 1e-1,    # 参数更新范数收敛阈值
        'bias_change': 1e-1      # 偏置项更新范数收敛阈值
    }
    
    # 动态获取参数数量
    num_params = len(gradient.para)

    # --- 2. 初始化优化器状态和历史记录 ---
    # Adam 算法所需的一阶矩（动量）和二阶矩（速度）向量
    first_moment = np.zeros(num_params, dtype=np.float64)
    second_moment = np.zeros(num_params, dtype=np.float64)
    
    # 特定于 QAOA 变体 ('ab') 的偏置项矩向量
    first_moment_bias = np.zeros(gradient.n, dtype=np.float64)
    second_moment_bias = np.zeros(gradient.n, dtype=np.float64)

    # 实例化 QAOA 对象
    qaoa = QAOA(gradient.n, gradient.qaoa_variant, gradient.pro_solve)
    
    # 初始化用于存储每次迭代结果的数组
    # 数组大小为 max_iters + 1，以存储初始状态 (第0次迭代)
    max_iters = int(gradient.max_iters)
    param_history = np.zeros(shape=(max_iters + 1, num_params), dtype=np.float64)
    bias_history = np.zeros(shape=(max_iters + 1, gradient.n), dtype=np.float64)
    energy_history = np.zeros(max_iters + 1, dtype=np.float64)
    bias_energy_history = np.zeros(max_iters + 1)
    gradient_history = np.zeros(shape=(max_iters + 1, num_params), dtype=np.float64)

    # --- 3. 设置初始状态 (t=0) ---
    param_history[0] = gradient.para.copy()
    
    # 计算初始梯度和能量
    initial_gradient, initial_energy = gradient_function(param_history[0], qaoa, _ADAM_HYPERPARAMS['gradient_precision'])
    gradient_history[0] = initial_gradient
    energy_history[0] = initial_energy
    
    # 记录初始偏置状态
    bias_history[0] = qaoa.bias
    bias_energy_history[0] = qaoa.get_bias_energy()

    # --- 4. 开始主循环 ---
    final_iteration = 0
    for t in range(1, max_iters + 1):
        final_iteration = t # 记录当前迭代次数

        # --- 4.1 Adam 参数更新 ---
        # 更新一阶矩（动量），g[t-1] 是上一步的梯度
        first_moment = _ADAM_HYPERPARAMS['beta1'] * first_moment + (1 - _ADAM_HYPERPARAMS['beta1']) * gradient_history[t-1]
        
        # 更新二阶矩（速度）
        second_moment = _ADAM_HYPERPARAMS['beta2'] * second_moment + (1 - _ADAM_HYPERPARAMS['beta2']) * (gradient_history[t-1] ** 2)
        
        # 对学习率进行偏差修正，以解决初始阶段矩估计向零偏差的问题
        corrected_learning_rate = _ADAM_HYPERPARAMS['learning_rate'] * np.sqrt(1 - _ADAM_HYPERPARAMS['beta2']**t) / (1 - _ADAM_HYPERPARAMS['beta1']**t)
        
        # 计算参数的更新量
        param_update = corrected_learning_rate * first_moment / (np.sqrt(second_moment) + _ADAM_HYPERPARAMS['epsilon'])
        
        # 更新参数
        param_history[t] = param_history[t-1] - param_update

        # --- 4.2 特定QAOA变体的偏置项更新 ---
        if qaoa.variant.get('type') == 'ab':
            bias_corrected_learning_rate = qaoa.variant['ell'] * np.sqrt(1 - _ADAM_HYPERPARAMS['beta2']**t) / (1 - _ADAM_HYPERPARAMS['beta1']**t)
            first_moment_bias, second_moment_bias = qaoa.update_bias(
                first_moment_bias, second_moment_bias, bias_corrected_learning_rate, param_history[t-1]
            )

        # --- 4.3 记录当前状态 ---
        bias_history[t] = qaoa.bias
        bias_energy_history[t] = qaoa.get_bias_energy()
        
        # 计算新参数下的梯度和能量
        current_gradient, current_energy = gradient_function(param_history[t], qaoa, _ADAM_HYPERPARAMS['gradient_precision'])
        gradient_history[t] = current_gradient
        energy_history[t] = current_energy

        # --- 4.4 检查收敛条件 ---
        energy_is_converged = np.abs(energy_history[t] - energy_history[t-1]) < _CONVERGENCE_THRESHOLDS['energy']
        params_are_converged = np.linalg.norm(param_update) < _CONVERGENCE_THRESHOLDS['param_change']
        bias_is_converged = np.linalg.norm(bias_history[t] - bias_history[t-1]) < _CONVERGENCE_THRESHOLDS['bias_change']
        
        if energy_is_converged and params_are_converged and bias_is_converged:
            break

    # --- 5. 封装并返回结果 ---
    # 使用 `final_iteration` 来截取有效的历史数据
    effective_iterations = final_iteration + 1
    other_history = {
        'theta': param_history[:effective_iterations],
        'energy_opt': energy_history[:effective_iterations],
        'bias_energy': bias_energy_history[:effective_iterations],
        'bias_opt': bias_history[:effective_iterations]
    }

    return GradientOutput(
        para_opt=param_history[final_iteration],
        energy=energy_history[final_iteration],
        itera=effective_iterations,
        bias_energy=bias_energy_history[final_iteration],
        bias_opt=bias_history[final_iteration],
        others=other_history
    )