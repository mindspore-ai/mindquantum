mindquantum.algorithm.compiler.u3_decompose
=============================================

.. py:function:: mindquantum.algorithm.compiler.u3_decompose(gate: U3)

    将U3门分解为Z-X-Z-X-Z旋转序列。

    分解遵循以下等式:
    U3(θ,φ,λ) = Rz(φ)Rx(-π/2)Rz(θ)Rx(π/2)Rz(λ)

    当任何旋转角度为常数且等于0时，相应的RZ门将被省略。

    参数：
        - **gate** (:class:`~.core.gates.U3`) - 需要被分解的U3门。

    返回：
        Circuit，使用ZXZXZ序列实现U3门的量子线路。
