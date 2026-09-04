mqvector_npu 单卡 NPU 测试与性能报告
=====================================

报告更新日期：2026-08-25

本报告说明 MindQuantum 实验性 ``mqvector_npu`` 后端的设计方案、代码结构、正确性测试、
性能 benchmark、优化效果和后续改进方向。该后端基于 CANN / Ascend C 实现单卡全振幅
状态向量模拟，当前定位为可评审的实验后端。

任务目标
--------

本任务要求为 MindQuantum 增加一个昇腾 NPU 模拟后端，并完成以下交付：

- 基于 CANN / Ascend C 实现 ``complex64`` 状态向量核心更新 kernel。
- 支持常见单比特门和双比特门，至少覆盖 ``X/Y/Z/H/RX/RY/RZ/CNOT/CZ``。
- 接入 MindQuantum Python API，支持创建模拟器、执行线路并获取最终态。
- 与 ``mqvector`` CPU 后端对齐小规模随机线路和典型线路的正确性。
- 给出不同量子比特数、不同量子门类型和不同线路深度下的 benchmark 数据。
- 分析核心 kernel 的访存开销、并行粒度、tiling 策略、优化效果和后续方向。

环境概览
--------

- Hardware: single-card NPU runtime
- CANN：``8.5.1``
- 操作系统：openEuler Linux
- 测试范围：单卡 NPU
- 状态向量精度：``complex64``
- CPU 对照后端：``mqvector``，``dtype=complex64``
- NPU 后端：``mqvector_npu``

代码结构
--------

- ``ccsrc/python/simulator/lib/_mq_vector_npu.cpp``

  - 管理 ACL runtime 生命周期、device memory、stream 和 Python 绑定。
  - 将 MindQuantum gate 转换为 NPU kernel 使用的紧凑 ``CircuitGate`` 描述。
  - 实现线路融合、local 分片、虚拟 ``SWAP`` 跟踪和线路调度。

- ``ccsrc/python/simulator/lib/mqvector_npu_kernel.cpp``

  - 实现 Ascend C device kernel。
  - 提供单比特矩阵门、双量子比特门、线路融合 kernel 和 local tile cache kernel。

- ``mindquantum/simulator/mqvector_npu.py``

  - 提供 Python 侧 ``MQVectorNPU`` 后端封装。
  - 约束当前仅支持 ``complex64``，并对未实现接口给出明确异常。

- ``mindquantum/simulator/available_simulator.py``

  - 探测当前构建是否支持 ``mqvector_npu``。

- ``tests/st/test_simulator/test_mqvector_npu.py``

  - 提供 NPU 后端专项正确性和边界测试。

- ``scripts/benchmark/benchmark_mqvector_npu.py``

  - 提供单卡可复现 benchmark，覆盖量子比特规模、门类型、线路深度和性能验收随机线路。

实现范围
--------

当前 NPU 后端支持 ``complex64`` 状态向量模拟。已实现门集合如下：

- ``X``
- ``Y``
- ``Z``
- ``H``
- ``RX``
- ``RY``
- ``RZ``
- ``PS``
- ``CNOT``
- ``CZ``
- ``SWAP``
- ``Rxx``
- ``Ryy``
- ``Rzz``

当前明确不支持的能力包括 measurement、sampling、expectation、gradients、
Hamiltonian evolution、BLAS helpers、circuit matrix、ground-state solving 和部分量子比特
reduced state extraction。这些接口会给出明确异常，避免静默返回不完整结果。

核心设计
--------

状态向量采用交错 ``complex64`` 布局，每个振幅占 8 bytes。一个 ``n`` qubit 状态需要
``8 * 2^n`` bytes 设备内存。单比特矩阵门以 target qubit 对应的 amplitude pair 为基本更新单元；
双量子比特门以对应的四个 basis amplitude 为基本更新单元。

host 侧将 Python 线路解析为固定 ABI 的 ``CircuitGate`` 数组，并一次性拷贝到 device 侧。
device kernel 按线路分片执行：每个分片选择一组 local qubit，把相关状态块搬入片上缓存，
连续执行该分片内的门后再写回全局状态。该设计减少逐门 launch/synchronize，也减少同一
local target 集合上的重复全局访存。

当前默认优化包括：

- 线路级 kernel 融合，减少逐门 kernel launch 和 stream 同步。
- local circuit fusion，将可放入同一 local qubit 集合的门合并到同一分片执行。
- local tile cache，默认最多缓存 ``14`` 个 local qubit。
- ``X/Y/Z/H/RX/RY/RZ/PS`` 和 ``SWAP/Rxx/Ryy/Rzz`` 专用计算路径。
- 虚拟 ``SWAP`` 跟踪，对无控制 ``SWAP`` 更新 qubit 映射，并在线路末尾恢复最终 qubit 顺序。
- 独立门调度，在保持语义正确的前提下重排可交换门，提升分片内门复用度。
- 26 qubit 默认使用 ``4096`` 的 NPU kernel blockDim；更小规模默认使用 ``2048``。

相关环境变量：

- ``MQVECTOR_NPU_FUSE_CIRCUIT=0``：关闭线路融合，回退到逐门路径。
- ``MQVECTOR_NPU_LOCAL_FUSION=0``：关闭 local circuit fusion。
- ``MQVECTOR_NPU_LOCAL_TILE_CACHE=0``：关闭 local tile cache。
- ``MQVECTOR_NPU_TRACK_SWAPS=0``：关闭虚拟 ``SWAP`` 跟踪。
- ``MQVECTOR_NPU_SCHEDULE_CIRCUIT=0``：关闭线路调度。
- ``MQVECTOR_NPU_LOCAL_CACHE_QUBITS=<value>``：设置 local cache qubit 数，上限为 ``14``。
- ``MQVECTOR_NPU_LEGACY_BLOCK_DIM=<value>``：覆盖默认 blockDim。

构建与运行
----------

在已安装 CANN 并可见单卡 NPU 的环境中，可使用如下方式构建和验证。路径请按实际环境替换：

.. code-block:: bash

   source /path/to/Ascend/ascend-toolkit/set_env.sh
   export MQ_OPENBLAS_LIB=/path/to/scipy_openblas32/lib/libscipy_openblas.so
   export MQ_OPENBLAS_SYMBOL_PREFIX=scipy_
   export LD_LIBRARY_PATH="$(dirname "${MQ_OPENBLAS_LIB}"):${LD_LIBRARY_PATH}"
   cmake -S . -B /path/to/build-dir \
       -DMQVECTOR_NPU_SOC_VERSION=target_soc
   cmake --build /path/to/build-dir --target _mq_vector_npu

专项测试命令：

.. code-block:: bash

   pytest -q tests/st/test_simulator/test_mqvector_npu.py

benchmark 命令：

.. code-block:: bash

   python scripts/benchmark/benchmark_mqvector_npu.py \
       --output /path/to/mqvector_npu_benchmark.json

性能验收随机线路命令：

.. code-block:: bash

   python scripts/benchmark/benchmark_mqvector_npu.py \
       --acceptance-only \
       --random-qubits 16,20,24,26 \
       --seed 20260821 \
       --output /path/to/mqvector_npu_acceptance.json

正确性测试结果
--------------

专项测试结果：

.. code-block:: text

   pytest -q tests/st/test_simulator/test_mqvector_npu.py
   29 passed

覆盖项包括：

- 后端注册和 ``complex64`` 精度约束。
- ``X/Y/Z/H/RX/RY/RZ/PS/CNOT/CZ`` 基础门。
- ``SWAP/Rxx/Ryy/Rzz`` 双量子比特门。
- Bell、GHZ 等典型线路。
- 固定 seed 的小规模随机线路。
- 非相邻 target qubit 和受控门。
- ``copy``、``astype(complex64)``、``reset`` 和完整状态读取。
- ``complex128``、部分量子比特状态读取和未实现能力的异常边界。
- 包含多次 ``SWAP``、受控门和双量子比特旋转门的虚拟 ``SWAP`` 跟踪回归线路。

所有 CPU/NPU 对照均使用 ``mqvector`` 的 ``complex64`` 结果作为参考，误差阈值为
``atol=1e-5``、``rtol=1e-5``。

补充 Benchmark
--------------

本节数据用于覆盖任务要求中的不同量子比特数、不同量子门类型和不同线路深度。表中耗时单位为 ms，
统计口径为 2 次预热、5 次正式运行的中位数；``apply`` 覆盖 ``reset + apply_circuit``，
``read`` 覆盖后续 ``get_qs``。

量子比特规模
~~~~~~~~~~~~

固定混合线路深度为 ``32``。

.. list-table::
   :header-rows: 1

   * - backend
     - qubits
     - apply
     - read
   * - ``mqvector``
     - 16
     - 305.622
     - 194.325
   * - ``mqvector_npu``
     - 16
     - 41.953
     - 0.106
   * - ``mqvector``
     - 20
     - 365.476
     - 295.670
   * - ``mqvector_npu``
     - 20
     - 654.305
     - 1.508
   * - ``mqvector``
     - 24
     - 902.154
     - 1292.274
   * - ``mqvector_npu``
     - 24
     - 10702.173
     - 47.549

门类型
~~~~~~

固定为 ``8`` qubit、深度 ``32``。``CNOT`` 使用公共 API ``CNOT.on(target, control)``。

.. list-table::
   :header-rows: 1

   * - gate
     - CPU apply
     - NPU apply
     - CPU read
     - NPU read
   * - ``X``
     - 0.011
     - 0.267
     - 0.013
     - 0.009
   * - ``Y``
     - 0.012
     - 0.265
     - 0.014
     - 0.009
   * - ``Z``
     - 0.010
     - 0.265
     - 0.013
     - 0.009
   * - ``H``
     - 0.017
     - 0.271
     - 0.014
     - 0.009
   * - ``RX``
     - 0.025
     - 0.265
     - 0.014
     - 0.009
   * - ``RY``
     - 0.024
     - 0.268
     - 0.014
     - 0.010
   * - ``RZ``
     - 0.024
     - 0.268
     - 0.014
     - 0.010
   * - ``CNOT``
     - 0.012
     - 0.188
     - 0.014
     - 0.009
   * - ``CZ``
     - 0.011
     - 0.247
     - 0.013
     - 0.009

线路深度
~~~~~~~~

固定为 ``8`` qubit 混合门线路。

.. list-table::
   :header-rows: 1

   * - depth
     - CPU apply
     - NPU apply
     - CPU read
     - NPU read
   * - 8
     - 0.007
     - 0.092
     - 0.014
     - 0.010
   * - 16
     - 0.010
     - 0.146
     - 0.013
     - 0.009
   * - 32
     - 0.017
     - 0.260
     - 0.014
     - 0.010
   * - 64
     - 0.030
     - 0.473
     - 0.013
     - 0.010

性能验收随机线路
----------------

性能验收负载使用 ``mindquantum.utils.random_circuit`` 构造：

.. code-block:: python

   random_circuit(n, n * 100, sd_rate=0.5, ctrl_rate=0.2, seed=seed)

测试使用 ``n = 16,20,24,26``，2 次预热、5 次正式运行。表中耗时单位为 ms，数据为均值。

.. list-table::
   :header-rows: 1

   * - qubits
     - gate_num
     - seed
     - CPU apply
     - CPU read
     - NPU apply
     - NPU read
     - apply 加速比
     - 总体加速比
   * - 16
     - 1600
     - 20261121
     - 4037.456
     - 81.863
     - 63.814
     - 0.103
     - 63.269x
     - 64.448x
   * - 20
     - 2000
     - 20261122
     - 17219.593
     - 241.825
     - 673.427
     - 1.527
     - 25.570x
     - 25.871x
   * - 24
     - 2400
     - 20261123
     - 27612.327
     - 1278.363
     - 11275.182
     - 48.903
     - 2.449x
     - 2.551x
   * - 26
     - 2600
     - 20261124
     - 51331.582
     - 4522.804
     - 48381.304
     - 192.514
     - 1.061x
     - 1.150x

在上述验收负载中，``16``、``20``、``24``、``26`` qubit 四组线路的 NPU ``apply`` 均快于 CPU；
算上完整状态读回后，NPU 总体耗时也均低于 CPU。

性能分析
--------

访存开销
~~~~~~~~

- 状态向量大小随 ``2^n`` 增长，26 qubit 的 ``complex64`` 状态约为 ``512 MiB``。
- 单比特门的基础更新单元是两个复振幅；未融合时每个门都会对相关 amplitude pair 做全局读写。
- local tile cache 将同一分片内的状态块复用到片上缓存，减少多门连续执行时的重复全局访存。
- ``get_qs`` 需要把完整状态向量从 device 拷回 host，耗时主要与状态向量大小相关。

并行粒度
~~~~~~~~

- kernel 以 amplitude pair 或四振幅块作为并行更新单元。
- local 分片路径通过 blockDim 分配不同 tile，26 qubit 默认使用 ``4096``。
- 每个分片内部按门序执行，分片之间通过 host 侧 launch 和 stream 同步保证状态依赖。

Tiling 策略
~~~~~~~~~~~

- host 侧根据当前分片的 target/control mask 选择 local qubit 集合。
- local tile cache 默认最多覆盖 ``14`` 个 local qubit。
- 独立门调度会优先选择能复用当前 local target 集合的门，降低分片切换次数。
- 对角门 ``Z/RZ/PS/Rzz`` 在调度中按可交换门处理，以增加可合并空间。

主要瓶颈
~~~~~~~~

- 26 qubit 已经达到 NPU 快于 CPU，但 ``apply`` 加速比仍较小，性能裕量主要取决于分片质量和
  大状态向量访存效率。
- 小规模、浅深度线路中，NPU 固定 launch/synchronize 开销占比较高，CPU 仍可能更快。
- 受控门仍需要处理控制 mask 过滤，控制条件不满足的 amplitude pair 会带来额外判断成本。
- 当前只支持完整状态读回，无法通过部分读回降低 host 侧输出成本。

结论
----

当前 ``mqvector_npu`` 已完成实验后端的核心交付：

- 可在 CANN / Ascend C 环境中编译和运行。
- 支持 ``complex64`` 全振幅状态向量模拟。
- 支持不少于 8 类基础量子门，并覆盖参数化旋转门和双量子比特门。
- 可通过 MindQuantum Python API 创建模拟器、执行线路并获取最终态。
- 专项正确性测试通过，CPU/NPU 结果在合理误差范围内一致。
- benchmark 覆盖不同量子比特数、门类型、线路深度和性能验收随机线路。
- 在 ``16/20/24/26``、``gate_num=n*100`` 随机线路上，NPU ``apply`` 均快于 CPU。

后续可改进方向
--------------

- 扩展 measurement、sampling、expectation 和 gradients 等完整模拟器能力。
- 继续优化 26 qubit 及更大规模下的分片策略和片上缓存复用。
- 针对受控门进一步减少无效 amplitude pair 访问和控制 mask 判断开销。
- 补充更系统的性能回归矩阵，用于防止后续改动削弱当前验收负载上的性能优势。
