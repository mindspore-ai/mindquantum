# 含噪条件下的变分量子算法求H4分子基态

完整仓库: https://github.com/Kahsolt/Mindquantum-Hackathon-2024

### problem

- 作答约束
  - 实现 [solution.py](solution.py) 脚本，入口函数签名 `solution(molecule, Simulator: HKSSimulator) -> float`
  - 限制使用门集合 {X,CNOT,Y,Z,H,CZ,RX,RY,RZ,Measure,Barrier}
  - 限制测量方式为求 pauli 串的哈密顿量期望
  - 已知噪声模型
- 评测条件
  - 提交: 仅 solution.py 文件
  - 用例: 各种键长的 `H4` 分子
  - 评分: L1_err↓, n_shots↓
  - 限时: 2h
- 考点
  - pauli grouping/trimming
    - 删减，尽可能少测几个pauli串
    - 分组，尽可能少测几次
    - 不同coeff的项测不同次数，以平衡coeff数值精度
    - X/Y轴投影测量会引入额外旋转门，可以想办法用Z轴投影去估算
      - 若用最基础的HF态线路，则只需测含Z的pauli串、且使用FIX_EXP，因为线路只含X门 (baseline)
  - light-weight ansatz design
    - 可能少的门，尽可能浅的线路
  - error mitigate
    - 去除空线路噪声
    - ZNE
    - 多测几次取最低
- 知识
  - $ H \left| \psi_0 \right> = \lambda_{min} \left| \psi_0 \right> $, 其中 $ \left| \psi_0 \right> $ 是 $ H $ 最小特征值 $ \lambda_{min} $ 所对应的特征向量 = 系统 $ H $ 的基态
  - 哈密顿量期望 $ \left< \psi_0 | H | \psi_0 \right> $ = 最小特征值 $ \lambda_{min} $ = 基态能量 $ E $
  - $ E $ 的误差来源
    - H 的近似处理
    - VQE 线路制备出来的不是 H 的基态本征态 $ \left| \psi_0 \right> $，能量会偏大
    - 测量的精度有限 0.01 ~ 0.0001

### solution

⚠ only run on Linux, due to dependecies of `openfermion` and `openfermionpyscf`

- run `python solver.py`

⚪ baselines

> truth $ E_{fci} = -2.166387 $, $ E_{hf} = -2.098546 $ for the default H4 1-2-3-4

| method | score↑ (clean/noisy) | energy↓ (clean/noisy) | time (clean/noisy) | comment |
| :-: | :-: | :-: | :-: | :-: |
| UCC |  47.484/0.609 | -2.14533/-0.52303 | 54.19/306.66 | baseline, init=zeros, shots=100 |
| UCC |  49.446/0.580 | -2.18661/-0.44119 | 57.59/635.20 | baseline, init=zeros, shots=1000 |
| UCC | 180.318/0.538 | -2.16084/-0.30754 | 55.81/307.34 | trim coeff < 1e-3 (184->180) |
| UCC |  22.286/0.596 | -2.12152/-0.48931 | 51.42/272.67 | trim coeff < 1e-2 (184->164) |

⚪ submits

| datetime | local score↑ | submit score↑ | comment |
| :-: | :-: | :-: | :-: |
| 2024-05-01 20:38:16 | 1.035 | 0.5226 | baseline, trim coeff < 1e-3, shot=30 |
| 2024-05-02 23:20:24 | 1.334 | 0.574  | ry_HEA, trim coeff < 1e-3, shots=100 |
| 2024-05-02 23:28:58 | 2.360 | 0.7994 | HF, trim coeff < 1e-3, shots=100 |
| 2024-05-05 23:15:30 | 2.360 | 1.0077 | HF, trim coeff < 1e-3, shots=100 |
| 2024-05-09 17:57:33 | 5.512 | 5.9368 | HF, trim coeff < 1e-3, shots=100, n_meas=10 |
| 2024-05-09 19:28:05 | 3.908 | 3.826  | ry_HEA, trim coeff < 1e-3, shots=100, n_meas=10 |
| 2024-05-09 19:28:05 | 4.519 | 4.2442 | ry_HEA_no_HF, trim coeff < 1e-3, shots=100, n_meas=10 |
| 2024-05-14 13:09:50 | 6~63 | 27.9413 (刷出来最高分) | HF, Z_only, shots=10, n_meas=10 |
| 2024-05-15 23:02:32 | 14.740 (非常固定) | 15.4219 | HF, Z_only (+exp_fix), shots=100 (**exactly $E_{HF}$**) |
| 2024-05-16 23:19:04 | 15.317 | 17.0523 | ry_HEA, depth=3 (2/4 都不好), shots=100 |
| 2024-05-16 23:21:33 |  8.426 | 13.4319 | ry_HEA, depth=3, shots=500 |
| 2024-05-16 23:24:39 | 17.266 | 19.3466 | ry_HEA, depth=3, shots=1000 |
| 2024-05-18 16:21:57 | 16.958 | 18.6189 | ry_HEA, depth=3, shots=3000 |
| 2024-05-18 23:48:44 | 7.924~87.441 | 9.6134~25.224 | ry_HEA, depth=3, init=randn, optim=Adam, shots=1000 |
| 2024-06-04 00:05:02 | 9~33 | 15.9935~47.1661 | HEA(RY), depth=3, init=randn, optim=Adam, shots=10000, combine_XY, rescaler |
| 2024-06-04 16:44:04 | 8.038~26.389 | 32.0759~127.9223 | HEA(RY), depth=2, init=randn, optim=Adam, shots=1000, combine_XY, rescaler |
| 2024-06-04 16:44:04 | 203.663 | 12.5186~320.646~3472.2812 | HEA(RY), depth=3, init=pretrained, optim=Adam(lr=1e-4), shots=1000, combine_XY, rescaler |

### reference

- solution from Tencent Quantum Lab: [https://github.com/liwt31/QC-Contest-Demo](https://github.com/liwt31/QC-Contest-Demo)
- mindquantum#error_mitigation: [https://gitee.com/mindspore/mindquantum/tree/error_mitigation/](https://gitee.com/mindspore/mindquantum/tree/error_mitigation/)
- mitiq: [https://mitiq.readthedocs.io/en/stable/](https://mitiq.readthedocs.io/en/stable/)
- Jordan–Wigner transformation: [https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation](https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation)
 - [https://futureofmatter.com/assets/fermions_and_jordan_wigner.pdf](https://futureofmatter.com/assets/fermions_and_jordan_wigner.pdf)

----
by Armit
2024/04/30
