# 基于QAIA的信号解码

完整仓库: https://github.com/Kahsolt/Mindquantum-Hackathon-2024

### problem

- 作答约束
  - 实现 [main.py](main.py) 脚本，入口函数签名
    - 构造 Ising 问题: `ising_generator(H:ndarray, y:ndarray, num_bits_per_symbol:int, snr:float) -> Tuple[ndarray, ndarray]`
    - 求解 Ising 问题: `qaia_mld_solver(J:ndarray, h:ndarray) -> ndarray`
- 评测条件
  - 评分: 比特错误率 BER↓, 运行时间 time↓
  - 限时: 1h
- 考点
  - MLD 问题转换为 Ising 问题
  - 改进 QAIA 算法，例如基于 SB 系列发展，或混合各类算法
  - 基线代码已经实现 `arXiv:2105.10535 [5]`，应该是希望选手进一步实现 `arXiv:2306.16264 [6]`

### solution

```
- baseline: bSB, B=100, n_iter=100
- found that LQA is better: LQA, B=100, n_iter=100
  - score: 0.7834 -> 0.7969
- add lmmse-regularization: LM-bSB, B=100, n_iter=100
- rescale reg-term with the best λ: LM-bSB, B=100, n_iter=100, λ=25
  - score: 0.7969 -> 0.8136
========================================================================================
the DU trick does not improve any, stuck here and then the scoring formula changed :)
========================================================================================
-> dramatically reduce iterations and repeatition: LM-bSB, B=1, n_iter=10, λ=25
  - time: 234s -> 5s
  - score: 33.8763
-> the DU trick imporves a little under this circumstance: DU-LM-bSB, B=1, n_iter=10
  - BER: 0.21313 -> 0.20111
-> conduct program-level optimization with profiling tools
  - time: 5s -> 3s
-> pReg: more flexible parametrization
  - BER: 0.20111 -> 0.15602
-> ppReg: even more flexible parametrization
  - BER: 0.15602 -> 0.15490
  - time: 3s -> 1.3s
```

### submits

⚪ baselines (classical)

- `pip install sionna`
- run `python run_baseline.py`

| method | BER↓ | comment |
| :-: | :-: | :-: |
| linear-zf-maxlog    | 0.41121 | match with the reference :) |
| linear-zf-app       | 0.40605 |  |
| linear-mf-maxlog    | 0.34104 |  |
| linear-mf-app       | 0.33401 |  |
| linear-lmmse-maxlog | 0.20779 |  |
| linear-lmmse-app    | 0.20721 | very fast; app > maxlog, lmmse > mf > zf |
| kbest-k=16          | 0.27594 | very slow |
| kbest-k=32          | 0.26785 |  |
| kbest-k=64          | 0.26206 |  |
| ep-iter=5           | 0.16968 | fast and nice! |
| ep-iter=10          | 0.16872 |  |
| ep-iter=20          | 0.16887 |  |
| ep-iter=40          | 0.16889 |  |
| mmse-iter=1         | 0.19903 | mmse is all cheaty, only for seeking BER lower bound!! |
| mmse-iter=2         | 0.16875 |  |
| mmse-iter=4         | 0.15921 |  |
| mmse-iter=8         | 0.15738 |  |
| mmse-iter=16        | 0.15768 |  |

⚪ baselines

- run `python judger.py`

| method | BER↓ | time | comment |
| :-: | :-: | :-: | :-: |
| ZF     | 0.41120 | -      | reference |
| NMFA   | 0.38561 | 230.66 |  |
| SimCIM | 0.23271 | 232.51 |  |
| CAC    | 0.31591 | 279.79 |  |
| CFC    | 0.23801 | 279.84 |  |
| SFC    | 0.23796 | 278.40 |  |
| ASB    | 0.34054 | 253.43 | dt=0.1 (default dt=1 doesn't run) |
| DSB    | 0.28741 | 230.11 |  |
| BSB[1] | 0.21584 | 135.38 | baseline[5] (B=100, n_iter=100) |
| LQA[2] | 0.20627 | 229.35 | best, but too classical |
| LM-bSB-λ=25  [6] | 0.18591 | 237.92 |  |
| DU-LM-SB-T=30[6] | 0.19497 |  76.91 | overfit(?) |

⚪ submits

| datetime | local BER↓ | submit score↑ | comment |
| :-: | :-: | :-: | :-: |
| 2024-04-30 21:12:31 | 0.20400 | 0.7969 | LQA (B=300) |
| 2024-05-02 21:25:46 | 0.21442 | 0.7834 | baseline (B=300) |
| 2024-05-06 11:37:08 | 0.18597 | 0.8136 | LM-bSB-λ=25 |
| 2024-05-21 18:08:50 | 0.39097 | 0.6089 | baseline (B=1, n_iter=1) |

#### After scoring formula updated (2024/5/21)

ℹ 评分公式更新为 $ \mathrm{score} = (1 - \mathrm{BER}) \times \frac{baseline\_time}{running\_time} $，实测只有在 BER 低于某阈值 (基线?) 的时候才会乘以时间项进行放大 😈

| method | BER↓ | time↓ | local score↑ | submit score↑ | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| BSB   | 0.21630 | 234.09 |  0.7837 |         | B=100, n_iter=100 |
| BSB   | 0.28433 |   1.16 |144.2781 |         | B=1, n_iter=10 |
| BSB   | 0.39115 |   2.87 | 49.7056 |  0.6092 | B=1, n_iter=1 |
| LM_SB | 0.18656 |  36.39 |  5.2332 |  4.9567 | B=10, n_iter=100; lowest BER |
| LM_SB | 0.18866 |  17.70 | 10.7307 |         | B=3, n_iter=100 |
| LM_SB | 0.18867 |  11.71 | 16.2184 |         | B=1, n_iter=100 |
| LM_SB | 0.19008 |  19.40 |  9.7726 |         | B=10, n_iter=50 |
| LM_SB | 0.19102 |  11.81 | 16.0343 |         | B=4, n_iter=50 |
| LM_SB | 0.19238 |   8.82 | 21.4311 | 20.3597 | B=1, n_iter=50 |
| LM_SB | 0.19658 |   6.93 | 27.1295 |         | B=1, n_iter=30 |
| LM_SB | 0.21313 |   5.27 | 34.9513 | 33.8763 | B=1, n_iter=10 |
| **LM_SB** | 0.21345 | 2.72 | 67.5861 | 92.4872 | B=1, n_iter=10 (after profiling); highest score (non-DU-method) |
| LM_SB | 0.22977 |   5.09 | 35.4083 |  0.7706 | B=1, n_iter=8 |
| LM_SB | 0.27122 |   4.89 | 34.9137 |         | B=1, n_iter=4 |
| LM_SB | 0.35360 |   5.34 | 28.3109 |         | B=1, n_iter=1 |
| DU_LM_SB | 0.22334 | 2.75 | 66.0949 |  | B=1, n_iter=4, lr=0.0001 |
| DU_LM_SB | 0.20696 | 2.65 | 70.0280 |  | B=1, n_iter=6, lr=0.0001 |
| **DU_LM_SB** | 0.21805 | 1.37 | 133.5814 | 105.2462 | B=1, n_iter=6, lr=0.0001, approx |
| DU_LM_SB | 0.20132 | 2.76 | 66.5661 |  | B=1, n_iter=8, lr=0.0001 |
| DU_LM_SB | 0.28677 | 5.59 | 29.8412 |  | B=1, n_iter=10, lr=0.01 |
| DU_LM_SB | 0.20071 | 3.36 | 55.6981 |  | B=1, n_iter=10, lr=0.0001 (after profiling) |
| DU_LM_SB | 0.20907 | 1.54 | 120.5792 | 101.1801 | B=1, n_iter=10, lr=0.0001, optim_steps=100000 (after profiling), approx |
| DU_LM_SB | **0.19932** | 2.73 | 68.5260 | 89.3023 | B=1, n_iter=10, lr=0.0001, optim_steps=100000 (after profiling) |
| DU_LM_SB | 0.20111 | 5.73 | 32.6491 |  | B=1, n_iter=10, lr=0.01, overfit |
| DU_LM_SB | 0.21522 | 8.50 | 21.6114 |  | B=1, n_iter=10, lr=0.01, overfit, update_hard |
| DU_LM_SB | 0.21898 | 6.85 | 26.6877 |  | B=1, n_iter=10, lr=0.01, overfit, use essay c_0 |
| DU_LM_SB | 0.20650 | 6.87 | 27.0227 |  | B=1, n_iter=10, lr=0.001, overfit |
| DU_LM_SB | 0.24786 | 5.37 | 32.7820 |  | B=1, n_iter=8, lr=0.01 |
| DU_LM_SB | 0.27747 | 2.74 | 61.8127 |  | B=1, n_iter=10, lr=0.01 (after profiling) |
| DU_LM_SB | 0.20085 | 3.32 | 56.4051 | 87.0405 | B=1, n_iter=10, lr=0.01, overfit (after profiling) |
| DU_LM_SB | 0.20049 | 2.89 | 64.8713 | 73.7797 | B=2, n_iter=10, lr=0.01, overfit (after profiling) |
| DU_LM_SB | 0.20022 | 2.99 | 62.5865 |  | B=3, n_iter=10, lr=0.01, overfit (after profiling) |
| DU_LM_SB | 0.20046 | 3.07 | 61.0498 |  | B=4, n_iter=10, lr=0.01, overfit (after profiling) |
| pReg_LM_SB |  |  |  |  | B=1, n_iter=6, lr=0.0001, optim_steps=50000 |
| **pReg_LM_SB** | 0.19940 | 2.77 | 67.6817 | 92.9875 | B=1, n_iter=10, lr=0.0001, optim_steps=100000; highest score (DU-method, non-cheaty) |
| pReg_LM_SB | 0.20015 | | | | B=1, n_iter=10, lr=0.0001, optim_steps=50000, refined from `DU-LM-SB_T=10_lr=0.0001.pth` |
| pReg_LM_SB | 0.21701 | 2.95 | 62.0686 | 89.8824 | B=1, n_iter=10, lr=0.01 |
| pReg_LM_SB | 0.15602 | 3.60 | 54.9289 | 95.1233 | B=1, n_iter=10, lr=0.01, overfit |
| pReg_LM_SB | 0.15584 | 3.66 | 54.0479 |  | B=2, n_iter=10, lr=0.01, overfit |
| pReg_LM_SB | 0.15545 | 4.66 | 42.3943 |  | B=3, n_iter=10, lr=0.01, overfit |
| ppReg_LM_SB | 0.2901 | 1.23 | 135.6155 |  | B=1, n_iter=10, lr=0.01 |
| ppReg_LM_SB | 0.2422 | 1.40 | 126.5427 |  | B=1, n_iter=10, lr=0.0001, optim_steps=100000 (after profiling) |
| **ppReg_LM_SB** | 0.15490 | 1.26 | 156.9497 | 137.3652 | B=1, n_iter=10, lr=0.01, overfit; highest score (DU-method, cheaty sense) |
| pppReg_LM_SB | 0.27084 | 1.25 | 136.5777 | | B=1, n_iter=10, lr=0.01 |
| pppReg_LM_SB | 0.20343 | 1.25 | 149.3551 | | B=1, n_iter=10, lr=0.01, overfit |

### dataset

```
[H] {(64, 64): 75, (128, 128): 75}
[y] {(64, 1): 75, (128, 1): 75}
[bits] {(64, 4): 30, (128, 6): 30, (64, 8): 15, (128, 8): 15, (128, 4): 30, (64, 6): 30}
[num_bits_per_symbol] {4: 60, 6: 60, 8: 30}
[SNR] {10: 50, 15: 50, 20: 50}

BER wrt. each param groups under latest best method:
>> avgber_per_Nt:
  64: 0.18860243055555556
  128: 0.18335503472222223
>> avgber_per_snr:
  10: 0.26107421875
  15: 0.18069010416666667
  20: 0.116171875
>> avgber_per_nbps:
  4: 0.10494791666666667
  6: 0.21564670138888892
  8: 0.2887044270833333
>> time cost: 335.17
>> avg. BER = 0.18598
```

### reference

The story timeline:

```
[2001.04014] Leveraging Quantum Annealing for Large MIMO Processing in Centralized Radio Access Networks (2020)
    - QuAMax + Quantum Anealing (D-Wave) + 4x4 16-QAM
    - the init SA impl, handle small H case
[2105.10535] Ising Machines' Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection
    - RI-MIMO/TRIM + CIM + 64x64 16-QAM (err: ~0.3)
    - the init QAIA impl, add reg_term $x_{LMMSE}$, proper baseline
  -> [2210.14660] Simulated Bifurcation Algorithm for MIMO Detection
      - RI-MIMO + dSB (G-SB) + 64x64 16-QAM (err: ~0.11)
      - upgrade CIM to dSB
    -> [2306.16264] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection
        - LM-SB/DU-LM-SB + bSB + 32x32 QPSK/4-QAM (err: ~1e-6??)
        - use LM algo, add DU
  -> [2304.12830] Uplink MIMO Detection using Ising Machines: A Multi-Stage Ising Approach
      - MDI-MIMO + CIM + 32x32 256-QAM (err: ~1e-4??)
      - multi-stage iterative of
          1. set reference solution $x_{MMSE}$ and $x_{MMSE-SIC}$ as init guess $x^*$
          2. search better results in a subspace $D_R$ around $x^*$
```

- [1] High-performance combinatorial optimization based on classical mechanics (2021): [https://www.researchgate.net/publication/349022706_High-performance_combinatorial_optimization_based_on_classical_mechanics](https://www.researchgate.net/publication/349022706_High-performance_combinatorial_optimization_based_on_classical_mechanics)
- [2] Quadratic Unconstrained Binary Optimization via Quantum-Inspired Annealing (2022): [https://www.researchgate.net/publication/363382279_Quadratic_Unconstrained_Binary_Optimization_via_Quantum-Inspired_Annealing](https://www.researchgate.net/publication/363382279_Quadratic_Unconstrained_Binary_Optimization_via_Quantum-Inspired_Annealing)
- [3] Leveraging Quantum Annealing for Large MIMO Processing in Centralized Radio Access Networks (2020): [https://arxiv.org/abs/2001.04014](https://arxiv.org/abs/2001.04014)
- [4] Physics-Inspired Heuristics for Soft MIMO Detection in 5G New Radio and Beyond (2021): [https://arxiv.org/abs/2103.10561](https://arxiv.org/abs/2103.10561)
- [5] Ising Machines' Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection (2021): [https://arxiv.org/abs/2105.10535](https://arxiv.org/abs/2105.10535)
- [6] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection (2023): [https://arxiv.org/abs/2306.16264](https://arxiv.org/abs/2306.16264)
- [7] Uplink MIMO Detection using Ising Machines: A Multi-Stage Ising Approach (2023): [https://arxiv.org/abs/2304.12830](https://arxiv.org/abs/2304.12830)
- [8] Simulated Bifurcation Algorithm for MIMO Detection (2022): [https://arxiv.org/abs/2210.14660](https://arxiv.org/abs/2210.14660)
- [9] Sionna: library for simulating the physical layer of wireless and optical communication systems
  - repo: [https://github.com/NVlabs/sionna](https://github.com/NVlabs/sionna)
  - doc: [https://nvlabs.github.io/sionna/index.html](https://nvlabs.github.io/sionna/index.html)
- [10] DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems (2022): [https://arxiv.org/abs/2212.07816](https://arxiv.org/abs/2212.07816)
  - repo: [https://github.com/IIP-Group/DUIDD](https://github.com/IIP-Group/DUIDD)

----
by Armit
2024/04/30
