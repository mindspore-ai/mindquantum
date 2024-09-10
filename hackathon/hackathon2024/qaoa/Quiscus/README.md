# 标准QAOA线路的最优初始化参数

完整仓库: https://github.com/Kahsolt/Mindquantum-Hackathon-2024

### problem

- 作答约束
  - 实现 [main.py](main.py) 脚本，入口函数签名 `main(Jc_dict:Dict[Tuple[int], float], p:int, Nq:int=12) -> Tuple[ndarray, ndarray]`
  - 禁止用目标函数对参数进行迭代优化 (i.e. non-adaptive)
- 评测条件
  - 用例: 随机 2~4 阶 Ising 模型，深度为 p=4/8 的标准 QAOA 线路
  - 评分: 线路末态哈密顿量的期望 C_d↓
  - 限时: 30min
- 考点
  - 量子逻辑线路初参选取，优化问题初始解选取
  - 递归/动态规划思想，大规模问题局部复用小规模问题的初始化参数
  - 预训练知识，总体规律
    - beta ↘: H0/H_B related
    - gamma ↗: H1/H_C related
  - 打一套精度更高的表 / 表生成算法
    - beta: 看似已经最优，不用做任何处理
    - gamma: 讨论放缩系数 (逐层不同，condition到问题配置)

### solution

- run `python score.py`

⚪ submits

ℹ Possible max local score: 23654.99239

| datetime | local score↑ | submit score↑ | ratio | comment |
| :-: | :-: | :-: | :-: | :-: |
| 2024-05-01 17:02:50 | 11730.14583 | 16627.496  | 1.418 | PT-WMC[1], reference, use $ γ^{inf} $ in stead of $ γ^{median} $ |
| 2024-05-01 16:44:12 | 16526.79871 | 24999.9025 | 1.513 | PS-WP[2], baseline  |
| 2024-05-06 23:09:02 | 17816.62534 | 27217.3001 | 1.528 | baseline, rescaler=1.275 |
|                     | 18516.17254 |            | | opt-avg, iter=10000, rescaler=1.0 |
|                     | 16150.62844 |            | | opt-avg, iter=10000, test with rescaler=1.275 |
|                     | 18089.52401 |            | | ft; iter=100, rescaler=1.275 |
| 2024-05-07 18:28:59 | 18302.26568 | 27756.869  | 1.517 | ft; iter=200, rescaler=1.275 |
| 2024-05-07 18:42:49 | 18183.73663 | 27645.5604 | 1.520 | ft; iter=300, rescaler=1.275 |
|                     | 18181.98281 |            | | ft; iter=400, rescaler=1.275 |
| 2024-05-07 19:35:12 | 18535.15152 | 27998.4487 | 1.511 | ft; iter=500, rescaler=1.275 |
|                     | 18452.12593 |            | | ft; iter=500 |
|                     | 17510.16457 |            | | ft; iter=1000, rescaler=1.275 |
|                     | 18911.62258 |            | | ft-ada; iter=100, rescaler=1.275 |
|                     | 19611.28956 |            | | ft-ada; iter=300, rescaler=1.275 |
|                     | 20027.51415 |            | | ft-ada; iter=400, rescaler=1.275 |
|                     | 18695.67729 |            | | ft-ada; iter=500, rescaler=1.275 |
|                     | 18203.21215 |            | | ft-ada; iter=600, rescaler=1.275 |
|                     | 18746.89745 |            | | ft-ada; iter=800, rescaler=1.275 |
|                     | 18739.11156 |            | | ft-ada; iter=900, rescaler=1.275 |
|                     | 19269.59404 |            | | ft-ada; iter=1000, rescaler=1.275 |
|                     | 18414.86418 |            | | ft-ada; iter=1300, rescaler=1.275 |
| 2024-05-31 15:45:22 | 20489.27983 | 29920.0329 | 1.460 | ft-ada; iter=1400, rescaler=1.275 |
| 2024-05-31 15:11:41 | 20369.67411 | 29777.4787 | 1.462 | ft-ada; iter=1500, rescaler=1.275 |
|                     | 20330.73329 |            | | ft-ada; iter=1900, rescaler=1.275 |
|                     | 20116.10485 |            | | ft-ada; iter=2000, rescaler=1.275 |
|                     | 19226.65152 |            | | ft-ada-decay; iter=2000, rescaler=1.275 |
| 2024-06-07 08:58:31 | 20368.31217 | 29747.6604 | 1.452 | ft-ada-decay; iter=3000, rescaler=1.275 |
| 2024-06-07 08:59:38 | 20175.90654 | 29554.6544 | 1.453 | ft-ada-decay; iter=4000, rescaler=1.275 |
| 2024-06-03 13:15:20 | 20696.82719 | 30072.2855 | **1.465** | ft-ada-decay; iter=5000, rescaler=1.275 |
| 2024-06-09 14:49:42 | 20726.71913 | 30100.7901 | 1.460 | ft-ada-decay; iter=5500, rescaler=1.275 |
| 2024-06-03 12:40:58 | 20729.63455 | 30105.3792 | 1.452 | ft-ada-decay; iter=5600, rescaler=1.275 |
|                     | 20802.38676 |            | | ft-ada-decay; iter=6000, rescaler=1.275 |
|                     | 20775.81853 |            | | ft-ada-decay; iter=6100, rescaler=1.275 |
|                     | 20703.96950 |            | | ft-ada-decay; iter=6400, rescaler=1.275 |
|                     | 20711.00992 |            | | ft-ada-decay; iter=6500, rescaler=1.275 |
|                     | 20797.86140 |            | | ft-ada-decay; iter=6700, rescaler=1.275 |
|                     | 20783.31786 |            | | ft-ada-decay; iter=6900, rescaler=1.275 |
|                     | 20814.45155 |            | | ft-ada-decay; iter=7000, rescaler=1.275 |
|                     | 20856.19912 |            | | ft-ada-decay; iter=7100, rescaler=1.275 |
|                     | 20778.53995 |            | | ft-ada-decay; iter=7200, rescaler=1.275 |
|                     | 20792.42218 |            | | ft-ada-decay; iter=8000, rescaler=1.275 |
|                     | 20824.90335 |            | | ft-ada-decay; iter=9000, rescaler=1.275 |
|                     | 20855.28777 |            | | ft-ada-decay; iter=9300, rescaler=1.275 |
|                     | 20859.12707 |            | | ft-ada-decay; iter=9400, rescaler=1.1 |
|                     | 20996.19230 |            | | ft-ada-decay; iter=9400, rescaler=1.14 |
|                     | 21014.58179 |            | | ft-ada-decay; iter=9400, rescaler=1.15 |
|                     | 21023.73164 |            | | ft-ada-decay; iter=9400, rescaler=1.16 |
| 2024-06-26 22:23:39 | 21024.64967 | 30259.2272 | 1.439 | ft-ada-decay; iter=9400, rescaler=1.165 |
|                     | 21023.28070 |            | | ft-ada-decay; iter=9400, rescaler=1.17 |
|                     | 21016.05226 |            | | ft-ada-decay; iter=9400, rescaler=1.18 |
|                     | 21014.30511 |            | | ft-ada-decay; iter=9400, rescaler=1.2 |
|                     | 20984.50076 |            | | ft-ada-decay; iter=9400, rescaler=1.26 |
|                     | 20962.71068 |            | | ft-ada-decay; iter=9400, rescaler=1.27 |
| 2024-06-11 11:10:03 | 20948.34952 | 30332.7544 | 1.448 | ft-ada-decay; iter=9400, rescaler=1.275 |
|                     | 20928.00170 |            | | ft-ada-decay; iter=9400, rescaler=1.28 |
|                     | 20907.75227 |            | | ft-ada-decay; iter=9500, rescaler=1.275 |
|                     | 20863.27086 |            | | ft-ada-decay; iter=9600, rescaler=1.275 |
|                     | 20809.78942 |            | | ft-ada-decay; iter=9700, rescaler=1.275 |
|                     | 20827.82030 |            | | ft-ada-decay; iter=10000, rescaler=1.275 |
|                     | 18027.36405 |            | | opt-avg; iter=1000, rescaler=1.275 |
|                     | 19479.14675 |            | | ft-ada-decay-moment-fast; iter=1400, rescaler=1.275 |
|                     | 19926.67130 |            | | ft-ada-decay-moment-fast; iter=2000, rescaler=1.275 |
|                     | 19905.01599 |            | | ft-ada-decay-moment-fast; iter=3000, rescaler=1.275 |
|                     | 19995.16122 |            | | ft-ada-decay-moment-fast; iter=3700, rescaler=1.275 |
| 2024-06-22 16:06:33 | 20381.14635 | 29800.8173 | 1.462 | ft-ada-decay-moment-fast; iter=3800, rescaler=1.275 |
|                     | 20145.42843 |            | | ft-ada-decay-moment-fast; iter=3900, rescaler=1.275 |
|                     | 20159.19957 |            | | ft-ada-decay-moment-fast; iter=4000, rescaler=1.275 |
|                     | 20377.47218 |            | | ft-ada-decay-moment-fast; iter=4500, rescaler=1.275 |
|                     | 20326.20533 |            | | ft-ada-decay-moment-fast; iter=4600, rescaler=1.275 |
|                     | 20100.26353 |            | | ft-ada-decay-moment-fast; iter=4700, rescaler=1.275 |
|                     | 19755.98407 |            | | ft-ada-decay-moment-fast; iter=4900, rescaler=1.275 |
|                     | 20916.35393 |            | | ft-ada-decay-moment-fast_ft; iter=9500, rescaler=1.275 |
|                     | 20878.52841 |            | | ft-ada-decay-moment-fast_ft; iter=9600, rescaler=1.275 |
|                     | 21031.99966 |            | | ft-ada-decay-moment-fast_ft; iter=9700, rescaler=1.165 |
|                     | 20950.16834 |            | | ft-ada-decay-moment-fast_ft; iter=9700, rescaler=1.275 |
| 2024-06-26 22:18:44 | 21033.52542 | 30264.7276 | 1.439 | ft-ada-decay-moment-fast_ft; iter=9800, rescaler=1.165 |
|                     | 20931.95560 |            | | ft-ada-decay-moment-fast_ft; iter=9800, rescaler=1.275 |
|                     | 20907.45221 |            | | ft-ada-decay-moment-fast_ft; iter=9900, rescaler=1.275 |
|                     | 20906.39396 |            | | ft-ada-decay-moment-fast_ft; iter=10000, rescaler=1.275 |
|                     | 21220.08623 |            | | ft-ada-moment-fast_ft-ex; iter=9700, rescaler=1.165 |
|                     | 21257.74984 |            | | ft-ada-moment-fast_ft-ex; iter=9800, rescaler=1.165 |
| 2024-06-26 22:13:46 | 21259.99295 | 30466.5136 | 1.433 | ft-ada-moment-fast_ft-ex; iter=9900, rescaler=1.165 |
|                     | 21237.71234 |            | | ft-ada-moment-fast_ft-ex; iter=10000, rescaler=1.165 |
|                     | 21213.81352 |            | | ft-ada-moment-fast_ft-ex; iter=10100, rescaler=1.165 |
|                     | 21205.14650 |            | | ft-ada-moment-fast_ft-ex; iter=10200, rescaler=1.165 |
|                     | 21275.64840 |            | | ft-ada-moment-fast_ft-ex; iter=10500, rescaler=1.165 |
|                     | 21258.07299 |            | | ft-ada-moment-fast_ft-ex; iter=11000, rescaler=1.165 |
|                     | 21360.43342 |            | | ft-ada-moment-fast_ft-ex; iter=11500, rescaler=1.165 |
|                     | 21250.40372 |            | | ft-ada-moment-fast_ft-ex; iter=12000, rescaler=1.165 |
|                     | 21416.90515 |            | | ft-ada-moment-fast_ft-ex; iter=12200, rescaler=1.165 |
| 2024-06-25 20:28:32 | **21432.33415** | **30641.0100** | 1.430 | ft-ada-moment-fast_ft-ex; iter=12300, rescaler=1.165 |
|                     | 21403.94442 |            | | ft-ada-moment-fast_ft-ex; iter=12400, rescaler=1.165 |

### reference

- [1] Parameter Transfer for Quantum Approximate Optimization of Weighted MaxCut (2023): [https://arxiv.org/abs/2201.11785](https://arxiv.org/abs/2201.11785)
- [2] Parameter Setting in Quantum Approximate Optimization of Weighted Problems (2024): [https://arxiv.org/abs/2305.15201](https://arxiv.org/abs/2305.15201)
- [3] The Quantum Approximate Optimization Algorithm at High Depth for MaxCut on Large-Girth Regular Graphs and the Sherrington-Kirkpatrick Model (2022): [https://arxiv.org/abs/2110.14206](https://arxiv.org/abs/2110.14206)

----
by Armit
2024/04/30
