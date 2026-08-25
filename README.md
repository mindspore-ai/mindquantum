# 昇腾CANN算子实现启发式算法

## 项目介绍

使用 AscendC 实现了离散模拟分岔DSB和弹道模拟分岔BSB算法。该实现使用了一个自定义内核，它在设备上运行整个仿真循环，以最大限度地减少主机-设备通信开销。矩阵-向量乘法通过向量单元（Vector unit）处理，并采用分块（tiling）技术，以在统一缓冲区（Unified Buffer）内高效支持 N=1024 的问题规模。


## 目录结构

* kernel/kernel_sb_kernel.cpp: AscendC 内核实现。
* host/main.cpp: 主机应用程序，用于设置数据、启动内核和验证执行。
* run.sh: 用于编译和运行项目的脚本。
* test.py: 测试程序

## 实现细节

* **算法：** 内核执行 niter 次更新规则迭代。
  * Sign(x) 是按元素计算的。
  * J.dot(Sign(x)) 是通过行级循环，结合向量乘法和归约（Mul, ReduceSum）计算的。
  * 条件处理 (np.where) 是通过掩码（Abs、Sign、算术运算）实现的，以避免分支。
* **内存管理：** 内核使用了双缓冲概念（尽管对于 N=1024 在此简化为单缓冲区），并管理全局内存（GM）和统一缓冲区（UB）之间的数据移动。
* **性能：** 整个循环在 AI Core 上运行，避免了 100 次往返主机端的通信。

## 数据集

基准数据集不随仓库分发（已在 `.gitignore` 中忽略），运行基准测试前请先下载到 `data/` 目录：

* **Gset 最大割基准**（`test_gset.py` 使用）：从 <https://web.stanford.edu/~yyye/yyye/Gset/> 下载 `G22`、`G30`、`G50`、`G60`，放入 `data/gset/`：

  ```bash
  mkdir -p data/gset
  for g in G22 G30 G50 G60; do
      wget -P data/gset https://web.stanford.edu/~yyye/yyye/Gset/${g}
  done
  ```

* **QPLIB 基准**（`test_qplib.py` 使用）：从 QPLIB 官网 <https://qplib.zib.de/>（或 GitHub 镜像 <https://github.com/ralna/QPLIB>）下载以下编号实例的 `QPLIB_XXXX.qplib` 文件，放入 `data/`：

  3506、3565、3642、3650、3693、3705、3706、3738、3745、3822、3832、3838、3850、3852、3877、5721、5725、5755、5875、5881、5882、5909、5922

## 如何运行

可以执行提供的运行脚本来构建和执行示例：

```bash
# 激活CANN环境变量，示例
source /home/jenkins0/Ascend/cann-8.5.0/set_env.sh
```

```bash
# 根据系统环境，修改 run.sh 的 ASCEND_INSTALL_PATH
# 根据系统环境，修改 kernel/CMakeLists.txt 和 host/CMakeLists.txt 的 ASCEND_CANN_PACKAGE_PATH

bash run.sh
```

输出将显示向量 x 和 y 的初始状态和最终状态。

使用pybind11将kernel_sb模块封装为python接口
test.py调用昇腾CANN实现的接口，计算矩阵的最大割和能量值，并比对CPU端和NPU端的性能。
