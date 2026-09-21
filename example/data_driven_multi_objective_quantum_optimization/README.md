# Data-Driven Multi-Objective Quantum Optimization

本案例整理了论文 [*Progress on Data-Driven, Multi-Objective Quantum Optimization*](https://arxiv.org/abs/2512.11479) 中 Figure 4 和 Figure 5 的小规模算法流程，覆盖合金数据生成、因子分解机（Factorization Machine，FM）、QUBO 构建、模拟退火、主动学习、绘图和结果验收。

> **阶段说明**：当前版本保留迁移前的 **PyTorch 二阶 FM + LBFGS** 训练后端，并使用 D-Wave Neal 的经典模拟退火求解 QUBO；尚未实现 MindSpore FM，也尚未接入 MindQuantum 量子线路。

## 快速开始

1，CLI调用方式

以下命令均从本案例目录执行：

```powershell
cd example/data_driven_multi_objective_quantum_optimization
conda env create -f environment.yml
conda activate mindquantum-cgfm-example
jupyter lab main.ipynb
```

Notebook 默认设置 `RUN_SCALE = "quick"`，并同时运行 Figure 4 和 Figure 5。迁移前同一实现的历史运行时间约为 Figure 4 19 小时 16 分、Figure 5 5 小时 50 分，合计约 25 小时；实际时间取决于 CPU/GPU、PyTorch 和模拟退火性能。首次阅读或验证安装时，建议先把 Notebook 参数单元改为 `RUN_SCALE = "test"`。

也可以先运行最短命令行烟测：

```powershell
python -m utils.figure4_runner --output-dir outputs/figure4_test --preset test `
  --settings wo_cgfm w_cgfm --device cpu --resume
python -m utils.figure5_runner --output-dir outputs/figure5_test --preset test `
  --settings w_ddts wo_ddts --device cpu --resume
```

2，参考 main.ipynb 中的运行方法。

## 项目范围

本案例提交的是源码和可复现运行入口，不提交数据集、论文 PDF、历史实验结果、PNG、日志、checkpoint、manifest、summary 或 validation report。初始合金数据由 `utils/alloy_dataset_generator.py` 根据固定 seed 在本地生成，无需下载外部数据。

三个 preset 的用途不同：

| preset | 用途 | 是否可被严格验收器接受 |
| --- | --- | --- |
| `test` | 单元测试和端到端烟测，验证最短代码路径及产物结构 | 否 |
| `quick` | 本项目定义的 canonical 小规模复现，验证完整工作流和结果自洽性 | 是 |
| `paper` | 保留论文规模参数，供研究和资源评估参考 | 否 |

`quick` 通过只表示“数据生成 → FM → QUBO → SA → active learning → summary → 绘图”的小规模实现完整且输出自洽，不表示复现了论文的计算规模、运行效率、数值曲线或相对性能结论。`paper` 可以显式运行，但计算量很大，预计耗时几个月。

## 目录结构

```text
data_driven_multi_objective_quantum_optimization/
├── main.ipynb                 # 教程、运行、绘图和验收入口
├── README.md                  # 本文档
├── environment.yml            # Conda 环境
├── .gitignore                 # 排除所有生成产物
├── utils/
│   ├── alloy_dataset_generator.py
│   ├── experiment_config.py
│   ├── experiment_runtime.py
│   ├── fm_torch.py
│   ├── qubo_math.py
│   ├── figure4_*.py
│   ├── figure5_*.py
│   └── validate_reproduction.py
└── tests/                     # 迁移后的单元测试和端到端烟测
```

核心模块职责如下：

| 模块 | 职责 |
| --- | --- |
| `alloy_dataset_generator.py` | 生成四相合金组成并计算五个真实性质 |
| `experiment_config.py` | 定义两张 Figure 共用的编码、FM、SA 和 `QuboConfig` |
| `experiment_runtime.py` | 管理依赖、device、seed、日志、原子 JSON 和 manifest 元数据 |
| `fm_torch.py` | PyTorch 二阶 FM、Optuna、LBFGS、数据拆分和 FM-to-QUBO |
| `qubo_math.py` | 离散编码、CGFM 映射、约束 QUBO、Neal 求解、可行解筛选和失败诊断 |
| `figure4_*.py` | Figure 4 配置、编码策略、pipeline、输出、CLI 和绘图 |
| `figure5_*.py` | Figure 5 配置、scalarization、Pareto、pipeline、输出、CLI 和绘图 |
| `validate_reproduction.py` | 对 canonical `quick` summary 做离线严格验收 |

## 共同算法与数据契约

### 合金数据与优化目标

每个候选由四相体积分数描述，数据生成器负责归一化和性质计算。Figure 4 逐个优化五个目标；Figure 5 联合优化前三个目标。

| 字段 | 含义 | 优化方向 |
| --- | --- | --- |
| `kappa` | 热导率 | 最大化 |
| `E` | 弹性模量 | 最大化 |
| `rho` | 密度 | 最小化 |
| `delta_alpha` | 热膨胀失配 | 最小化 |
| `delta_T` | 与目标相变温度的偏差 | 最小化 |

FM/QUBO 内部统一按“越小越好”训练，因此最大化目标会先取负号；记录和绘图仍使用真实物理量及真实优化方向。目标在训练前进行 z-score 标准化。

### FM 训练规则

当前后端标识为 `pytorch_fm_lbfgs`。实现保留二阶 FM、rank 6、最多 2000 个训练步、Optuna 超参数搜索和 FM-to-QUBO 过程。论文补充材料使用 fastFM + ALS，两者可能形成不同的能量地形。

- 数据量不少于 5 时，显式拆分 train/validation/test，并保证三部分非空。
- Optuna trial 只使用 train/validation，以 `validation_loss` 选择超参数。
- test 集只在选定超参数后的最终拟合中评估。
- 数据量少于 5 时，三种用途共享当前小数据集，以支持最短烟测。
- FM-to-QUBO 会丢弃不影响最优 bit 状态的常数偏置 `w0`。

### QUBO 组合规则

完整 QUBO 由共享配置统一管理：

```text
QuboConfig
  fm_objective_weight: 1.0
  system_penalty_weight: 650.0
  one_hot_penalty_weight: 1.0
  normalization_scheme: max_abs_polynomial_coefficient_v1
```

组合公式为：

```text
Q = w_fm     * N(Q_fm)
  + w_system * N(Q_system)
  + w_onehot * N(Q_onehot)
```

`N` 使用 `max_abs_polynomial_coefficient_v1`：线性项取 `q[i,i]`，二次项取 `q[i,j] + q[j,i]`，常数项不参与尺度计算。Figure 4 的 `w_cgfm` 不需要 system term，因此实际 `system_penalty_weight` 和 `system_scale` 记录为 `0.0`。CLI 可通过 `--fm-objective-weight`、`--system-penalty-weight` 和 `--one-hot-penalty-weight` 覆盖相应权重。

共享 dataclass 和 `paper` preset 的 one-hot 权重为论文值 `1.0`；`quick`/`test` 显式使用 `2.0`，用于适配当前 PyTorch FM 能量地形。soft penalty 只改变能量，不保证有限数量的 SA reads 一定包含可行终态。

### SA 候选选择与失败诊断

Figure 4 和 Figure 5 共用以下规则：

1. 保留 Neal 返回的全部 reads，并按能量升序排列。
2. 依次解码，选择第一个同时满足编码约束和四相总和约束的候选。
3. 如果整批 reads 均不可行，才精确比较最低采样不可行能量与可行编码域最优能量。
4. 最多诊断 `1_000_000` 个可行状态；诊断不会修改实验结果。
5. 若最低不可行能量更低，报告 `QUBO penalty-balance failure`；若可行域最优能量更低，报告 `heuristic-sampling failure`；无法区分时报告原因未分类。
6. 无论分类为何，该轮都会明确失败，不会把精确枚举结果、cure 或随机状态冒充 QUBO 解。
7. 只有可行 SA proposal 与已有 composition 重复时，才生成唯一 random replacement。

summary 会记录 `infeasible_sa_samples_skipped`、`max_feasible_candidate_rank`、`duplicate_replacements`、`random_replacements` 和 `random_replacement_draws`，便于审计候选选择过程。配置字段使用 D-Wave 的 `reads` 术语，CLI 首选 `--sa-reads`；`--sa-runs` 仅作为兼容别名。

### 可复现性与恢复契约

- base seed 必须满足 `0 <= seed <= 2**31 - 1`。
- 数据、CGFM、SA、FM 和 replacement 使用确定且彼此隔离的随机流。
- Figure 5 同一 seed 下的两种 setting 有意共享初始数据和 preference weights，保证公平比较。
- FM 训练 seed 使用 NumPy seed 空间的高半区，避免与 Neal 的低半区 seed 冲突。
- runner 会把解析后的完整配置、seed 方案、依赖、CUDA 和 Git 状态写入 manifest。
- checkpoint 绑定 Figure、setting、objective（适用时）、seed、schema 和完整配置；身份不一致时拒绝恢复。

修改任何 preset 覆盖项后都应使用新的 `--output-dir`，不要把不同配置的 checkpoint 混在同一目录。

## Notebook 使用方法

`main.ipynb` 按以下顺序组织完整示例：

1. 项目背景和阶段说明。
2. Python、依赖和 CUDA 环境检查。
3. 统一设置 `RUN_SCALE`、Figure 开关、device 和 `outputs/` 路径。
4. 演示合金数据生成以及 FM/QUBO/CGFM 原理。
5. 通过 `python -m utils.figure4_runner` 和 `python -m utils.figure5_runner` 执行实验。
6. 读取普通 JSON summary，调用绘图模块并显示图片。
7. `quick` 执行严格 validator；`test` 只检查端到端流程与产物结构。

Notebook 提交时不保存 execution count、输出、图片或实验数据。运行产生的所有内容都写入被忽略的 `outputs/`。如果 CUDA 不可用，参数单元会回退到 CPU；命令行运行时则应显式传入可用的 `--device cpu` 或 `--device cuda`。

## 环境与测试

`environment.yml` 固定了经过迁移验证的主要依赖：Python 3.9、PyTorch 1.9.1、NumPy 1.26.4、SciPy 1.10.1、scikit-learn 1.6.1、Matplotlib 3.5.1、Optuna 4.8.0、Dimod 0.12.21 和 D-Wave Neal 0.6.0，并包含 Jupyter、ipykernel 和 nbconvert。

已有同名环境时可以更新：

```powershell
conda env update -n mindquantum-cgfm-example -f environment.yml --prune
conda activate mindquantum-cgfm-example
```

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

迁移基准为 124 项测试全部通过。还可以检查五个正式模块入口：

```powershell
python -m utils.figure4_runner --help
python -m utils.figure5_runner --help
python -m utils.plot_figure4 --help
python -m utils.plot_figure5 --help
python -m utils.validate_reproduction --help
```

## 第一部分：Figure 4 复现

### Figure 4 当前状态

- 已实现 `wo_cgfm`（直接四相编码）和 `w_cgfm`（CGFM 角度编码）。
- 已覆盖 `kappa`、`E`、`rho`、`delta_alpha` 和 `delta_T` 五个单目标任务。
- 已实现 PyTorch FM、FM-to-QUBO、SA 全 reads 筛选、重复候选替换、best-so-far 更新和多 seed 聚合。
- 已实现 per-trajectory checkpoint、严格恢复校验、manifest、日志、checkpoint schema v6 和 summary schema v5。
- `quick` 共运行 3 seed × 5 objective × 2 setting = 30 条 trajectory，每条 100 轮。

### Figure 4 代码结构

```text
utils/alloy_dataset_generator.py     合金组成和五个真实性质
utils/experiment_config.py           EncodingConfig/FMConfig/SAConfig/QuboConfig
utils/experiment_runtime.py          device、seed、I/O、日志和 manifest
utils/figure4_experiment_config.py   ObjectiveSpec、Figure4ExperimentConfig 和 preset
utils/figure4_setting_strategies.py  wo_cgfm/w_cgfm 编码与解码策略
utils/fm_torch.py                    PyTorch FM、Optuna、LBFGS 和 FM-to-QUBO
utils/qubo_math.py                   QUBO、CGFM、SA、可行性与诊断
utils/figure4_pipeline.py            trajectory、active learning、恢复与聚合
utils/figure4_outputs.py             Figure4Summary、checkpoint 和输出布局
utils/figure4_runner.py              正式 CLI 入口
utils/plot_figure4.py                summary 绘图
```

### Figure 4 算法流程

每个 seed 先生成一份由全部 objective/setting 共享的初始数据，然后分别执行以下过程：

1. 按 objective 方向转换目标并执行 z-score 标准化。
2. `wo_cgfm` 使用四个 one-hot 相组成 block；`w_cgfm` 使用三个 CGFM 角度 block。
3. 使用确定的 stage seed 搜索超参数并最终拟合 FM。
4. 将二阶 FM 转换为目标 QUBO，并与适用的 system/one-hot 约束组合。
5. 执行 Neal 模拟退火，按能量顺序选择第一个可行候选。
6. 重复 proposal 只触发唯一 random replacement；不会伪造新的 SA 解。
7. 计算候选真实性质、追加数据，并更新真实方向的 best-so-far。
8. 每轮原子写入 checkpoint；全部 trajectory 完成后聚合均值、总体标准差、最小值和最大值。

Figure 4 绘制各 seed 的历史最优值。曲线保持水平通常表示本轮候选没有改善历史最优，不等同于训练失败。

### Figure 4 对象命名与调用规则

| 概念 | 固定名称 |
| --- | --- |
| setting | `wo_cgfm`、`w_cgfm` |
| objective | `kappa`、`E`、`rho`、`delta_alpha`、`delta_T` |
| 完整配置 | `Figure4ExperimentConfig` |
| 汇总对象 | `Figure4Summary` |
| 正式入口 | `python -m utils.figure4_runner` |

Notebook 也只通过模块 CLI 调用完整流程，避免交互式工作目录和包内相对导入不一致。

### Figure 4 配置规则

| preset | seed | 初始样本 | 每条轨迹迭代 | levels | Optuna trials | SA reads | SA sweeps | one-hot 权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper` | 20 | 100 | 600 | 50 | 20 | 1000 | 3000 | 1.0 |
| `quick` | 3 | 50 | 100 | 25 | 3 | 250 | 1000 | 2.0 |
| `test` | 1 | 10 | 2 | 8 | 0 | 32 | 12 | 2.0 |

`quick` 下，`wo_cgfm` 为 100-bit QUBO，`w_cgfm` 为 75-bit QUBO；直接编码可表示 `C(28,3)=3276` 个可行组成。`paper` 的 50 levels 分别对应 200 bit 和 150 bit，计算代价显著更高。

所有数值参数都可以通过 CLI 覆盖，但覆盖后得到的是自定义运行，不再具有 canonical `quick` 身份。

### Figure 4 运行方法

先运行低成本烟测：

```powershell
python -m utils.figure4_runner --output-dir outputs/figure4_test --preset test `
  --settings wo_cgfm w_cgfm --device cpu --resume
python -m utils.plot_figure4 --summary outputs/figure4_test/figure4_summary.json `
  --output outputs/figure4_test/figure4.png
```

资源允许时运行 canonical `quick`：

```powershell
python -m utils.figure4_runner --output-dir outputs/figure4_quick --preset quick `
  --settings wo_cgfm w_cgfm --device cuda --resume
python -m utils.plot_figure4 --summary outputs/figure4_quick/figure4_summary.json `
  --output outputs/figure4_quick/figure4.png
```

如果 PyTorch CUDA 不可用，把 `--device cuda` 改为 `--device cpu`。

### Figure 4 输出规则

```text
outputs/figure4_quick/
├── figure4_manifest.json
├── figure4_summary.json
├── figure4.png
├── figure4_validation.json        # 执行严格验收后生成
├── logs/
│   ├── figure4_runner.log
│   └── plot_figure4.log
└── trajectories/
    └── {setting}_{objective}_seed_{seed}.json
```

summary 是一个普通 JSON 文档，不是 JSONL。可以直接检查其结构：

```powershell
python -m json.tool outputs/figure4_quick/figure4_summary.json
```

manifest、checkpoint、summary、validation、日志和 PNG 均是运行产物，已被案例级 `.gitignore` 排除，不应提交到 PR。

### Figure 4 验收规则

严格验收仅接受两种 setting、全部 objective 和 canonical 参数均完整的 `quick` summary：

```powershell
python -m utils.validate_reproduction figure4 `
  --summary outputs/figure4_quick/figure4_summary.json
```

validator 依次执行严格契约解析、Figure 自洽重建、非阻断 diagnostics 和 JSON-safe report；它不会重新训练 FM、重建 QUBO 或运行 SA。`test` summary 会被严格验收器按设计拒绝，烟测时只需确认 runner、summary、绘图和产物结构正常。

## 第二部分：Figure 5 复现

### Figure 5 当前状态

- 已实现 `w_ddts` 和 `wo_ddts`（weighted-sum baseline）两条多目标流程。
- 已实现最大化 `kappa`、最大化 `E`、最小化 `rho` 的统一方向处理。
- 已实现可复现 preference weights、DDTS 人工目标、三 FM QUBO 合并和 Pareto front。
- 已区分 proposed solution 与实际 added solution；重复 proposal 会保留，random replacement 不进入 Pareto front。
- 已实现 checkpoint schema v5、summary schema v5、恢复校验、manifest 和日志。

### Figure 5 代码结构

```text
utils/figure5_experiment_config.py  Figure5ExperimentConfig 和 preset
utils/figure5_scalarization.py      preference weights、DDTS 和 weighted-sum 参考目标
utils/figure5_pareto.py             Figure5ParetoFront 和非支配解筛选
utils/figure5_pipeline.py           多目标 active-learning trajectory 与聚合
utils/figure5_outputs.py            Figure5Summary、checkpoint 和输出布局
utils/figure5_runner.py             正式 CLI 入口
utils/plot_figure5.py               3D Pareto 和迭代窗口绘图
```

`compute_weighted_sum_reference_targets` 用于生成 weighted-sum 参考目标并支持 summary/validator 的一致性检查。

### Figure 5 算法流程

两种 setting 在相同 seed 下共享初始数据和每轮 preference weights，然后采用不同 surrogate 路径：

- `w_ddts`：把三个方向统一后的目标合成为一个 DDTS 人工目标，每轮拟合一个 FM，再求解对应 QUBO。
- `wo_ddts`：分别为三个 objective 拟合 FM，根据同一组 preference weights 合并三个 QUBO，作为 weighted-sum baseline。

每轮都记录 preference weights、proposal、实际加入的数据、FM/QUBO 元数据和 Pareto 信息。Pareto front 从真实 SA proposed solutions 构建；为保持数据唯一性而加入的 random replacement 不进入 Pareto 统计。

### Figure 5 对象命名与调用规则

| 概念 | 固定名称 |
| --- | --- |
| setting | `w_ddts`、`wo_ddts` |
| objective | `kappa`、`E`、`rho` |
| 完整配置 | `Figure5ExperimentConfig` |
| 汇总对象 | `Figure5Summary` |
| Pareto 对象 | `Figure5ParetoFront` |
| 正式入口 | `python -m utils.figure5_runner` |

### Figure 5 配置规则

| preset | seed | 初始样本 | 每条轨迹迭代 | levels | Optuna trials | SA reads | SA sweeps | one-hot 权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper` | 1 | 500 | 1000 | 25 | 20 | 1000 | 3000 | 1.0 |
| `quick` | 1 | 500 | 150 | 25 | 3 | 100 | 500 | 2.0 |
| `test` | 1 | 10 | 2 | 8 | 0 | 32 | 12 | 2.0 |

Figure 5 与 Figure 4 共用 FM、SA、seed schedule、`QuboConfig` 和 `max_abs_polynomial_coefficient_v1` 归一化契约。`quick` 共执行两条 setting trajectory、300 个 active-learning iteration；`wo_ddts` 每轮训练三个 FM，因而通常比 `w_ddts` 更耗时。

### Figure 5 运行方法

先运行低成本烟测：

```powershell
python -m utils.figure5_runner --output-dir outputs/figure5_test --preset test `
  --settings w_ddts wo_ddts --device cpu --resume
python -m utils.plot_figure5 --summary outputs/figure5_test/figure5_summary.json `
  --output outputs/figure5_test/figure5.png
```

资源允许时运行 canonical `quick`：

```powershell
python -m utils.figure5_runner --output-dir outputs/figure5_quick --preset quick `
  --settings w_ddts wo_ddts --device cuda --resume
python -m utils.plot_figure5 --summary outputs/figure5_quick/figure5_summary.json `
  --output outputs/figure5_quick/figure5.png
```

### Figure 5 输出规则

```text
outputs/figure5_quick/
├── figure5_manifest.json
├── figure5_summary.json
├── figure5.png
├── figure5_validation.json        # 执行严格验收后生成
├── logs/
│   ├── figure5_runner.log
│   └── plot_figure5.log
└── trajectories/
    └── {setting}_seed_{seed}.json
```

Figure 5 summary 区分 proposed solution、added solution、重复 proposal 和 random replacement。它同样是普通 JSON，不是 JSONL：

```powershell
python -m json.tool outputs/figure5_quick/figure5_summary.json
```

### Figure 5 验收规则

严格验收要求 canonical `quick` 的两个 setting 均存在，并检查 preference weights、目标方向、weighted-sum 参考值、Pareto front、QUBO 统计、checkpoint 审计字段和 summary 自洽性：

```powershell
python -m utils.validate_reproduction figure5 `
  --summary outputs/figure5_quick/figure5_summary.json
```

论文趋势以及 DDTS 覆盖率/均匀性只作为 diagnostics 写入报告，不影响 `passed`。`test` 模式仍只用于端到端烟测，不执行严格 reproduction validator。

## 恢复运行与问题诊断

两个 runner 都支持 `--resume`，并在每轮完成后写入 trajectory checkpoint。运行中断后，使用完全相同的命令即可恢复；Notebook 中重新执行相应运行单元格也会带上 `--resume`。

常见情况：

- **checkpoint 配置不匹配**：不要删除或强行复用，改用新的输出目录。
- **CUDA 不可用**：改为 `--device cpu`；runner 会在训练前检查 device。
- **整批 SA 无可行候选**：根据错误分类检查 penalty 配比，或调整 reads/sweeps/求解器；保持配置不变恢复会重现相同失败批次。
- **只有平台曲线**：先检查 best-so-far 定义、离散网格极值和候选审计字段，不能仅凭平台判断 FM 失败。
- **validator 拒绝 test/paper**：这是预期行为；严格 validator 只接受 canonical `quick`。

## 已知限制与后续工作

- 当前没有 `fm_mindspore.py`，训练逻辑仍在 `fm_torch.py`。
- 当前 QUBO 由 Neal 经典模拟退火求解，没有构造 MindQuantum 量子线路。
- `paper` 规模在普通开发机上成本很高，保留它主要是为了明确论文参数。
- `compute_delta_t` 按论文舍入系数实现；精确共晶点 `fSi = 0.128` 返回 0，但连续输入在其极窄邻域可能出现非物理负值。canonical `quick` 网格不会落入该区间，自定义 levels 时需要注意。
- 后续替换为 MindSpore FM 时，应尽量保持配置 dataclass、seed 方案、FM metadata、QUBO 归一化、checkpoint/summary schema 和五个 CLI 的外部契约不变，以便对比迁移前后行为。

