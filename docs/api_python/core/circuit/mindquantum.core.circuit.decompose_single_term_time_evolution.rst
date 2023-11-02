mindquantum.core.circuit.decompose_single_term_time_evolution
==============================================================

.. py:function:: mindquantum.core.circuit.decompose_single_term_time_evolution(term, para)

    将时间演化门分解成基本的量子门。

    这个函数只适用于只有单个泡利词的哈密顿量。
    例如， :math:`exp^{-it\text{ham}}` ， :math:`\text{ham}` 只能是一个泡利词，如 :math:`\text{ham}=X_0 Y_1 Z_2` 。此时，结果是 `((0, 'X'), (1, 'Y'), (2, 'Z'))`。
    当演化时间被表示成 :math:`t=ax+by` 时，参数将是 `{'x':a, 'y':b}`。

    参数：
        - **term** (tuple, QubitOperator) - 仅演化量子算子的哈密顿量项。
        - **para** (Union[dict, numbers.Number]) - 演化算子的参数。

    返回：
        Circuit，量子线路。

    异常：
        - **ValueError** - 如果 `term` 里有多个泡利句。
        - **TypeError** - 如果 `term` 不是 `map`。
