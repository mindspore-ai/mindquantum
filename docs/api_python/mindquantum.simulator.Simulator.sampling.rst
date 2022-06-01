.. py:method:: mindquantum.simulator.Simulator.sampling(circuit, pr=None, shots=1, seed=None)

    在线路中对测量比特进行采样。采样不会改变模拟器的量子态。

    **参数：**

    - **circuit** (Circuit) - 要进行演化和采样的电路。
    - **pr** (Union[None, dict, ParameterResolver]) - 线路的parameter resolver，如果线路是含参线路则需要提供pr。默认值：None。
    - **shots** (int) - 采样线路的次数。默认值：1。
    - **seed** (int) - 采样的随机种子。如果为None，则种子将是随机的整数。默认值：None。

    **返回：**

    MeasureResult，采样的统计结果。