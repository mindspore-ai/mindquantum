.. py:class:: mindquantum.core.circuit.Circuit(gates=None)

    量子电路模块。
    量子电路包含一个或多个量子门，可以在量子模拟器中进行计算。可以通过添加量子门或另一电路的方式容易地构建量子电路。

    **参数：**

    - **gates** (BasicGate, list[BasicGate]) - 可以通过单个量子门或门列表初始化量子电路。默认值：None。
