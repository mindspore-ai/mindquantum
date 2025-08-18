### Analysis

问题: 调整一组天线阵子的振幅 `A` 和相位 `φ`，可以让波束表现出目标形状 `S=f(A, φ)` 指向方向 `θ0`，这可以转为 QUBO 组合优化问题，即已知目标 `S/θ0` 希望用量子启发式 QAIA 优化方法求出 `A` 和 `φ`
  - 这是个简单的函数优化问题，其中相位 `φ` 取值离散 (1~4bit量化)，振幅 `A` 取值连续，两者都是长度 `N=32` 的向量
  - 考察点就是优化方法、损失函数设计、匠心调参
评分: `score = 1000 - 100*外旁瓣惩罚a - 80*主瓣惩罚b - 20*内旁瓣惩罚c`，单样例运行时间90s，超时0分
  - θ0: 测试用例的主瓣方向在 [45°,135°] 之间，优化后的主瓣峰值与目标必须相差 ≤1° 否则0分
  - a: 外旁瓣最大值低于主瓣 -15dB
  - b: 主瓣宽度 ≤6°
  - c: 内旁瓣最大值低于主瓣 -30dB

```
[理想波形]
  45°        135°
        θ0 (ε=1°)
        /\
___    /  \    ___  -15dB
   \__/    \__/     -30dB
      <-6°->
  -30°  0°   30°
```

- 基线方法
  - `answer_BSB_Adam.py` 先用 torch 实现的 SB 算法优化定 `φ`，再用 Adam 优化定 `A`
  - `answer_BSB.py` 用 mindquantum 实现的 SB 算法优化定 `φ`
  - 注: SB系列算法中的微分方程，`x` 为广义坐标，`y` 为广义动量


### Local

⚠ 算法输出有随机性！！

| method | time (s) | score |
| :-: | :-: | :-: |
| baseline   | 14.40~16.38 | 41.0138~142.8581 |
| baseline-1 | 8.719 | 0 |


### Submit

| method | score | datetime |
| :-: | :-: | :-: |
| baseline | 305.6669 | 2025-05-06 22:03:20 |
| my       | 219.0569 | 2025-05-17 17:00:17 |


#### Essay Notes

⚪ arXiv:2409.19938: Quantum-inspired Beamforming Optimization for Quantized Phase-only Massive MIMO Arrays

```
[proposed Algo]
- map quantized phases to a set of spin variables
  - a kind of gray code, pure phase-modulation
- formulate the Hamiltonian of the BF optimization (a quantum annealing process)
  - initial: nonlinear Kerr Hamiltonian
  - final: BF Hamiltonian
- run the quantum-inspired SB algorithm on a classical computer
  - SB is x100 faster than Genetic Aglo
```
