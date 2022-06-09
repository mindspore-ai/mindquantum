.. py:class:: mindquantum.core.operators.QubitExcitationOperator(term=None, coefficient=1.0)

    量子比特激发算子定义为： :math:`Q^{\dagger}_{n} = \frac{1}{2} (X_{n} - iY_{n})` 和 :math:`Q_{n} = \frac{1}{2} (X_{n} + iY_{n})` 。
    与费米子激发算子相比，量子比特激发算子是某种“局部化”的，即费米子激发算子 :math:`a^{\dagger}_{7} a_{0}` 涉及到JW变换下从0到7的量子比特，
    而量子比特激发 :math:`Q^{\dagger}_{7} Q_{0}` 只会影响第0和第7个量子位。
    此外，用量子比特激发算子描述双激发所使用的CNOT门比相应的费米子激发算子少得多。

    **参数：**

    - **terms** (str) - 量子比特激发算子的输入项。 默认值：None。
    - **coefficient** (Union[numbers.Number, str, ParameterResolver]) - 相应单个运算符的系数。默认值：1.0。
