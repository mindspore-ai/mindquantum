mindquantum.core.operators.QubitExcitationOperator
===================================================

.. py:class:: mindquantum.core.operators.QubitExcitationOperator(term=None, coefficient=1.0)

    量子比特激发算子定义为： :math:`Q^{\dagger}_{n} = \frac{1}{2} (X_{n} - iY_{n})` 和 :math:`Q_{n} = \frac{1}{2} (X_{n} + iY_{n})` 。
    与费米子激发算子相比，量子比特激发算子是某种“局部化”的，即费米子激发算子 :math:`a^{\dagger}_{7} a_{0}` 涉及到JW变换下从0到7的量子比特，
    而量子比特激发 :math:`Q^{\dagger}_{7} Q_{0}` 只会影响第0和第7个量子比特。
    此外，用量子比特激发算子描述双激发所使用的CNOT门比相应的费米子激发算子少得多。

    参数：
        - **terms** (Union[str, tuple]) - 量子比特激发算子的输入项。默认值： ``None``。
        - **coefficient** (Union[numbers.Number, str, ParameterResolver]) - 相应单个运算符的系数。默认值： ``1.0``。

    .. py:method:: hermitian()

        返回量子比特激发算子的厄米共轭。

    .. py:method:: imag
        :property:

        该算符的虚部。

        返回：
            QubitExcitationOperator，保留原始算符虚部的量子比特激发算符。

    .. py:method:: normal_ordered()

        按照比特序号由小到大排列量子比特激发算符。

        .. note::
            与费米子不同，玻色子交换不需要乘系数-1。

        返回：
            QubitExcitationOperator，正规排序后的量子比特激发算符。

    .. py:method:: real
        :property:

        该算符的实部。

        返回：
            QubitExcitationOperator，保留原始算符实部的量子比特激发算符。

    .. py:method:: to_qubit_operator()

        将量子比特激发算子转化为泡利算符。

        返回：
            QubitOperator，根据量子比特激发算符定义相对应的泡利算符。
