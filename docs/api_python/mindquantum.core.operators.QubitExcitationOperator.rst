.. py:class:: mindquantum.core.operators.QubitExcitationOperator(term=None, coefficient=1.0)

    Qubit Excitation Operator定义为：
    :math:`Q^{\dagger}_{n} = \frac{1}{2} (X_{n} - iY_{n})` 和 :math:`Q_{n} = \frac{1}{2} (X_{n} + iY_{n})`。
    与Fermion excitation operators相比，Qubit Excitation Operator是某种“局部化”，即，
    Fermion excitation operators :math:`a^{\dagger}_{7} a_{0}` 在JW变换下涉及的量子比特范围是0到7，而Qubit excitation :math:`Q^{\dagger}_{7} Q_{0}` 将仅影响第0和第7个量子比特。
    此外，使用Qubit excitation operators描述的双激发比相应的Fermion excitation operators使用的CNOT要少得多。

    **参数：**

    - **terms** (str) - qubit excitation operator的输入项。默认值：None。
    - **coefficient** (Union[numbers.Number, str, ParameterResolver]) - 对应单运算符的系数，默认值：1.0。    