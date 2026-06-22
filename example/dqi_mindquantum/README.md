# Decoded Quantum Interferometry (DQI) with MindQuantum

本案例展示了如何使用
[MindQuantum](https://www.mindspore.cn/mindquantum) 框架实现
**Decoded Quantum Interferometry (DQI)** 算法，
求解 max-XORSAT 组合优化问题。

**论文引用:**
Sabater, El Harzli, Besjes et al.
"Towards solving industrial integer linear programs with Decoded Quantum Interferometry."
arXiv:2509.08328 (2025).

## 算法简介

DQI 是一种量子优化算法，通过量子干涉和解码技术
（LUT 或 BP1）求解 max-XORSAT 问题：

1. 将组合优化问题编码为量子态的振幅
2. 通过多项式度 ℓ 控制采样偏置
3. 利用后选择（post-selection）提取高质量解

量子比特数为 O(m+n)，线性扩展，与 ℓ 无关。

## 目录结构

```text
dqi_mindquantum/
├── main.ipynb                    # 主入口
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖
├── classical/                    # 经典算法模块
│   ├── bp_decoders.py            # BP 解码器
│   ├── dqi_expectation.py        # DQI 期望计算
│   ├── matrix_generation.py      # 矩阵生成
│   └── random_sampling.py        # 随机采样基线
├── quantum/                      # 量子电路模块
│   ├── circuit_components.py     # 电路组件
│   ├── decoding.py               # 解码器电路
│   ├── dicke_states.py           # Dicke 态
│   ├── full_circuit.py           # DQICircuit 类
│   ├── unk_state.py              # 未知态构造
│   └── weight_encoding.py        # 权重编码
└── experiments/                  # 实验脚本
    ├── reproduce_fig11.py        # Fig 11 复现
    ├── reproduce_fig15.py        # Fig 15 复现
    └── scaling_analysis.py       # 资源分析
```

## 环境要求

- Python >= 3.11
- MindQuantum >= 0.9.0

## 快速开始

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## 数据说明

本案例不包含预计算数据。
所有矩阵均在运行时生成：

- `experiments.reproduce_fig11.generate_reference_matrix()`：生成参考 LDPC-like 矩阵
- `classical.matrix_generation.generate_downsized_matrices()`：基于度分布生成降采样矩阵

## 参考资源

- [MindQuantum 官方文档](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html)
- [论文 arXiv:2509.08328](https://arxiv.org/abs/2509.08328)
