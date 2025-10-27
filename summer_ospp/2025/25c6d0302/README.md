# QAOA 算法复现与性能分析项目结项报告

---

## 一、 项目基本信息

-   **项目名称**: 基于 MindQuantum 的高级 QAOA 变体算法复现与性能分析
-   **核心算法**: QAOA, Warm-Starting QAOA, Adaptive-Biasing QAOA (ab-WS-QAOA)
-   **目标问题**: Max-Cut (最大割问题)
-   **相关论文**: arXiv:2503.20048
-   **原项目链接**: [Yunlong350/WS-ab-QAOA](https://github.com/Yunlong350/WS-ab-QAOA)

## 二、 项目背景与目标

本项目旨在复现论文 `arXiv:2503.20048` 中提出的几种高级量子近似优化算法（QAOA）变体，并验证其在解决 Max-Cut 问题上的性能。我们利用华为 MindSpore Quantum (MindQuantum) 框架作为主要的量子计算模拟工具。

项目的主要目标包括：
1.  搭建可复现的实验环境，并实现标准 QAOA、温启动 QAOA (WS-QAOA) 和自适应偏置 QAOA (ab-WS-QAOA) 的核心逻辑。
2.  开发工具脚本，用于生成和获取 Max-Cut 问题的基准数据集，包括与 NEOS Server 交互以获得精确解。
3.  复现原论文中的核心对比图（图2, 3, 4, 5），通过图表直观展示不同算法变体在不同问题规模下的性能差异和优势。
4.  提供一个清晰、完整的代码库和说明文档，方便后续研究者学习和使用。

## 三、 项目成果

本项目成功完成了预定目标，主要成果如下：

1.  **算法实现**: 完整实现了三种 QAOA 算法。特别是对于核心的 `ab-WS-QAOA`，实现了论文中定义的特定 Adam 优化器来更新偏置场 `h`。
2.  **数据生成**: 实现了与 NEOS Server 的自动化通信，可获取小规模图的 Max-Cut 精确解作为基准。
3.  **结果复现**: 成功复现了原论文中的关键图表，验证了 `ab-WS-QAOA` 算法在运行初始阶段即能达到 Goemans-Williamson (GW) 算法的温启动效果，并在大规模问题上展现出性能优势。

## 四、 环境配置与使用说明

### 4.1 环境配置步骤

**前提**: 确保您的电脑已经安装了 Python 3.10。**注意：版本不宜过高**，否则 `QAOA_demo.py` 中的 MindSpore 函数可能无法正常调用。

1.  **解压项目**: 将 `QAOA_Project.zip` 文件解压到目标位置。
2.  **打开终端**:
    * **Mac**: 打开“终端” (Terminal) 应用。
    * **Windows**: 打开“命令提示符” (Command Prompt) 或 “PowerShell”。
3.  **进入项目目录**: 在终端中使用 `cd` 命令进入 `QAOA_Project` 文件夹。
    ```bash
    # 示例:
    cd /path/to/your/QAOA_Project
    ```
4.  **创建并激活Python虚拟环境**:
    * **Mac / Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    激活成功后，命令行提示符前会出现 `(venv)` 字样。
5.  **安装所有依赖**: 在激活的虚拟环境中，运行以下命令安装所有必需的库。
    ```bash
    pip install -r requirements.txt
    ```
    此过程可能需要几分钟。`pip` 会自动处理所有库的版本依赖。

### 4.2 如何运行

环境配置完成后，您可以直接运行主脚本来执行演示或复现图像：
```bash
# 运行一个6-qubit的小型演示
python QAOA_demo.py

# 运行图3的复现脚本
python Plot_fig3.py
```

## 五、 模块功能说明

### 5.1 `QAOA_demo.py`
-   **功能**: 使用 MindQuantum 通过高级 QAOA 变体解决 Max-Cut 问题。该脚本整合了标准 QAOA、温启动 QAOA 和自适应偏置 QAOA (ab-WS-QAOA)。由于 ab-WS-QAOA 的显著优势主要体现在大规模问题（如 n~1000）上，此脚本仅作为一个 6-qubit 的功能演示，用于检验 ab-WS-QAOA 在初始阶段即能达到 GW 温启动的切割边数。

### 5.2 `create_dataset.py`
-   **功能**: 与 NEOS Server 通信，调用其线上求解器以获取小规模 Max-Cut 问题的精确解。
-   **配置**: 运行前必须在脚本中配置个人信息。
    -   **前提**: 前往 [NEOS Server 官网](https://neos-server.org/neos/) 注册账号。
    -   **参数**:
        ```python
        MY_USERNAME = "your_username"          # 用户名
        MY_PASSWORD = "your_password"          # 密码
        MY_EMAIL = "your_email@example.com"    # 邮箱
        DATA_FILE = "test.dat"                 # 结果保存文件
        POLLING_DELAY_SECOND = 15              # 每15秒查询一次状态 
        MAX_WAIT_MINUTES = 20                  # 最长等待20分钟
        ```

### 5.3 `Plot_fig2.py`
-   **功能**: 复现论文中的图2。
-   **配置**:
    ```python
    # 要求此文件夹中至少包含n=40到n=140的Max-Cut数据集
    DATASET_DIR = "u3r_dataset"
    # 是否重新计算数据，False则直接读取已有数据绘图
    RECALCULATE = False
    # 默认数据读取/保存路径
    DATA_FILENAME = 'data/figure2_data.npz'
    # config参数控制数据生成范围
    'node_sizes': [40, 60, 80, 100, 120, 140] 
    'qaoa_initial_point': np.array([[5.7665, 4.4898]]),
    'qaoa_initial_point2': np.array([[4.2315, 1.0002]])
    ```

### 5.4 `Plot_fig3.py`
-   **功能**: 复现论文中的图3。
-   **配置**:
    ```python
    RECALCULATE = False
    DATA_FILENAME = 'data/figure3_data.npz'
    # 若 RECALCULATE = True，以下参数生效
    node_sizes = [200, 300, 400, 500, 600, 700, 800, 900, 1000] # 节点规模
    realizations = 40             # 每个规模下随机u3r图的数量
    R_gw = 1                      # GW 算法 random rounding 次数
    qaoa_initial_point = np.array([[4.2315, 1.0002]]) # p=1 QAOA 初始参数
    max_iters = 4000              # p=1 QAOA 最大迭代次数
    ```

### 5.5 `Plot_fig4.py`
-   **功能**: 复现论文中的图4。
-   **配置**:
    ```python
    RECALCULATE = True
    DATA_FILENAME = 'figure4_final_data.npz'
    config = {
        'n': 1000,                  # 比较 rounding 数影响时的节点数
        'realizations': 40,         # 抽样图的数量
        'qaoa_initial_point': np.array([[4.2315, 1.0002]]),
        'max_iters': 2000,
    }
    ```

### 5.6 `Plot_fig5.py`
-   **功能**: 复现论文中的图5。
-   **配置**:
    ```python
    RECALCULATE = False
    DATA_FILENAME = 'figure5_cumulative_data.npz'
    # 依赖图3已生成的数据
    FIG3_DATA_FILENAME = 'figure3_data.npz'
    # 若 RECALCULATE = True，以下参数生效
    config = {
        'node_sizes': [200, 400, 600, 800, 1000],
        'realizations': 40,
        # GW算法random rounding的次数列表
        'R_values': [0, 10, 50, 100, ..., 1000000], 
    }
    ```