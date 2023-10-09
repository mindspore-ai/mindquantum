mindquantum.algorithm.library.qft
==================================

.. py:function:: mindquantum.algorithm.library.qft(qubits)

    量子傅里叶变换（QFT）。量子傅里叶变换与经典傅里叶变换的功能相似。

    .. note::
        更多信息请参考Nielsen, M., & Chuang, I. (2010)。

    参数：
        - **qubits** (list[int]) - 需要应用量子傅里叶变换的量子比特。

    返回：
        Circuit，可以进行傅里叶变换的线路。
