mindquantum.algorithm.library.general_ghz_state
================================================

.. py:function:: mindquantum.algorithm.library.general_ghz_state(qubits)

    基于零态制备通用GHZ态的线路。
    GHZ态通常定义为三个全零态和三个全一态的均匀叠加：

    .. math::

        \left|\text{GHZ}\right> = (\left|000\right> + \left|111\right>)/\sqrt{2}

    在本接口中，我们可以创建出任意量子比特规模下任意部分量子比特之间的GHZ态。

    参数：
        - **qubits** (list[int]) - 需要应用通用GHZ态的量子比特。

    返回：
        Circuit，可以制备GHZ态的线路。
