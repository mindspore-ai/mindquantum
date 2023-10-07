mindquantum.algorithm.error_mitigation.zne
=====================================================

.. py:function:: mindquantum.algorithm.error_mitigation.zne(circuit: Circuit, executor: typing.Callable[[Circuit], float], scaling: typing.List[float] = None, order=None, method="R", a=0, args=None)

    零噪声外推算法。

    参数：
        - **circuit** (:class:`~.core.circuit.Circuit`) - 一段量子线路。
        - **executor** (Callable[[:class:`~.core.circuit.Circuit`], float]) - 一个可执行的接口，该接口可以演化一段量子线路，并返回一个值。
        - **scaling** (List[float]) - 线路折叠的系数。如果是 ``None``，该值将为 ``[1.0, 2.0, 3.0]`` 。默认值： ``None``。
        - **order** (float) - 多项式外推的指数。默认值： ``None``。
        - **method** (str) - 外推的方法，可以时 ``'R'`` （Richardson）， ``'P'`` （多项式）和 ``'PE'`` （指数多项式）。默认值： ``'R'``。
        - **a** (float) - 当方法时指数多项式时的度。默认值： ``0``。
        - **args** (Tuple) - ``executor`` 执行函数除第一个参数外的其他参数。
