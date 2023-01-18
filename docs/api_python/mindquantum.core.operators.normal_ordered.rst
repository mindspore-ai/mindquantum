mindquantum.core.operators.normal_ordered
==========================================

.. py:function:: mindquantum.core.operators.normal_ordered(fermion_operator)

    计算并返回FermionOperator的规范顺序。
    根据惯例，规范顺序意味着项从最高模式索引（左侧）到最低（右侧）排序。
    此外，创建运算符首先出现，然后跟随湮灭运算符。
    例如 '3 4^' :math:`\rightarrow` '- 4^ 3' 。

    参数：
        - **fermion_operator** (FermionOperator) - 此参数只能为费米子类型运算符。

    返回：
        FermionOperator, 规范有序的FermionOperator。
