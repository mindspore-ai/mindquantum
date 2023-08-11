mindquantum.simulator.fidelity
====================================

.. py:function:: mindquantum.simulator.fidelity(rho: numpy.ndarray, sigma: numpy.ndarray)

    计算两个量子态的保真度。

    参数：
        - **rho** (numpy.ndarray) - 其中一个量子态。支持态矢量或密度矩阵。
        - **sigma** (numpy.ndarray) - 另一个量子态。支持态矢量或密度矩阵。

    返回：
        numbers.Number，两个量子态的保真度。
