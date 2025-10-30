"""
Fig3(b)(c) another method 核心函数模块
使用SPSA方法的结构搜索训练
"""

import sys
import os
# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入基础函数
from fig3bc_functions import (
    hamiltonian_xxz,
    vqe_circuit,
    _single_qubit_layer,
    SHVQEEvaluator,
    ground_state_mq_order,
    all_code_probs_from_bits_prob,
    clifford_ti_from_code
)
from mindquantum.core.parameterresolver import PRGenerator
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def softmax2_rows(A: np.ndarray) -> np.ndarray:
    """计算2维softmax"""
    A = A - A.max(axis=1, keepdims=True)
    e = np.exp(A)
    return e / e.sum(axis=1, keepdims=True)


def alpha_to_p1(alpha: np.ndarray) -> np.ndarray:
    """从alpha计算每个距离的开启概率"""
    return softmax2_rows(alpha)[:, 1]


def sample_bits_with_crn(Urand: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """使用通用随机数采样bits"""
    return (Urand < p1[None, :]).astype(int)


def bits_to_codes(bits: np.ndarray) -> list:
    """将bits转换为代码列表"""
    return [''.join(map(str, row.tolist())) for row in bits]


def entropy_bernoulli(p1: np.ndarray) -> float:
    """计算伯努利分布的熵"""
    p1 = np.clip(p1, 1e-8, 1-1e-8)
    H  = -(p1*np.log(p1) + (1-p1)*np.log(1-p1))
    return float(np.sum(H))


class AdamOpt:
    """Adam优化器实现"""
    def __init__(self, size, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps
        self.m=np.zeros(size); self.v=np.zeros(size); self.t=0
    def step(self, x, g):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*g
        self.v = self.b2*self.v + (1-self.b2)*(g*g)
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return x - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def train_with_empirical_log(ev,
                     iters=800, nsamp=800,
                     warmup_iters=40, inner_steps=5,
                     spsa_c_th=0.10, spsa_c_ph=0.10, spsa_c_al=0.05,
                     spsa_exp=0.101,
                     lr_theta=0.02, lr_phi=0.02, lr_alpha=0.10,
                     tau0=0.05, tau_exp=0.5,
                     nsamp_log=200,
                     seed_train=7, seed_log=7 + 99991,
                     log_interval=20):
    """
    使用SPSA方法训练
    """
    rng_tr  = np.random.default_rng(seed_train)
    rng_log = np.random.default_rng(seed_log)
    k   = ev.n // 2
    all_codes = [format(i, f'0{k}b') for i in range(1 << k)]

    # 参数初始化
    theta = rng_tr.normal(0, 0.1, len(ev.theta_names))
    phi   = rng_tr.normal(0, 0.1, len(ev.phi_names))
    alpha = np.zeros((k, 2), dtype=float)

    opt_th = AdamOpt(len(theta), lr=lr_theta)
    opt_ph = AdamOpt(len(phi),   lr=lr_phi)
    opt_al = AdamOpt(alpha.size, lr=lr_alpha)

    logs = {'cost': [], 'p_bits': [], 'p_codes': []}

    def energy_mean(theta, phi, alpha, Urand):
        p1    = alpha_to_p1(alpha)
        codes = bits_to_codes(sample_bits_with_crn(Urand, p1))
        return ev.mean_energy_over_codes(codes, theta, phi)

    for t in range(1, iters + 1):
        # 本轮 CRN（训练）
        Urand = rng_tr.random(size=(nsamp, k))

        # SPSA 步长与熵系数
        ck_th = spsa_c_th / (t ** spsa_exp)
        ck_ph = spsa_c_ph / (t ** spsa_exp)
        ck_al = spsa_c_al / (t ** spsa_exp)
        tau   = (tau0 / (t ** tau_exp)) if t > warmup_iters else 0.0

        # θ,φ 内层
        inner_loops = inner_steps if t > warmup_iters else 1
        for _ in range(inner_loops):
            dth = rng_tr.choice([-1.0, 1.0], len(theta))
            dph = rng_tr.choice([-1.0, 1.0], len(phi))
            Jp  = energy_mean(theta + ck_th*dth, phi + ck_ph*dph, alpha, Urand)
            Jm  = energy_mean(theta - ck_th*dth, phi - ck_ph*dph, alpha, Urand)
            g_scalar = (Jp - Jm) / (2.0 * max(ck_th, ck_ph))
            theta = opt_th.step(theta, g_scalar * dth)
            phi   = opt_ph.step(phi,   g_scalar * dph)

        # α 外层
        dal = rng_tr.choice([-1.0, 1.0], alpha.size).reshape(alpha.shape)
        Jp  = energy_mean(theta, phi, alpha + ck_al*dal, Urand)
        Jm  = energy_mean(theta, phi, alpha - ck_al*dal, Urand)
        Jp += tau * entropy_bernoulli(alpha_to_p1(alpha + ck_al*dal))
        Jm += tau * entropy_bernoulli(alpha_to_p1(alpha - ck_al*dal))
        g_al = (Jp - Jm) / (2.0 * ck_al)
        alpha = opt_al.step(alpha.ravel(), g_al * dal.ravel()).reshape(alpha.shape)

        # 记录
        p1_now = alpha_to_p1(alpha)
        J_now  = energy_mean(theta, phi, alpha, Urand)

        Ulog = rng_log.random(size=(nsamp_log, k))
        bits_log  = sample_bits_with_crn(Ulog, p1_now)
        codes_log = bits_to_codes(bits_log)
        cnt = Counter(codes_log)
        probs = {c: cnt.get(c, 0)/nsamp_log for c in all_codes}

        logs['cost'].append(J_now)
        logs['p_bits'].append(p1_now.copy())
        logs['p_codes'].append(probs)

    return {'theta': theta, 'phi': phi, 'alpha': alpha, 'logs': logs}


def plot_fig3b_spsa(logs):
    """绘制SPSA方法的Fig3b结果"""
    # 上图：Cost
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1, len(logs['cost'])+1), logs['cost'], linewidth=1.6)
    plt.xlabel('Iteration'); plt.ylabel('Cost (Energy)')
    plt.title('Fig.3(b) Upper: Cost vs Iteration')
    plt.tight_layout(); plt.show()

    # 下图：代码概率
    plt.figure(figsize=(6,3))
    codes_draw = sorted(logs['p_codes'][-1].keys())
    for code in codes_draw:
        ys = [d[code.translate({48:49, 49:48})] for d in logs['p_codes']]
        plt.plot(ys, label=code, linewidth=1.0)
    plt.legend(ncol=4, fontsize=8, frameon=False)
    plt.xlabel('Iteration'); plt.ylabel('Probability (empirical)')
    plt.title('Fig.3(b) Lower: Code probabilities')
    plt.tight_layout(); plt.show()


def plot_fig3c_from_results(evaluator, codes_array, theta, phi):
    """绘制Fig3c的fidelity柱状图"""
    fid_array = np.array([evaluator.fidelity_with_ground_state(code, theta, phi) for code in codes_array])
    
    x = np.arange(len(codes_array))
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(x, fid_array, color="#f5c08f", edgecolor="#8a5d29", linewidth=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(codes_array, fontsize=10)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, 1.0)
    
    ax.set_xlabel("Graph Connection type")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fig.3(c): Final fidelity by graph type")
    
    plt.tight_layout()
    return fig

