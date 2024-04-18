mindquantum.algorithm.library.mat_to_op
=======================================================

.. py:function:: mat_to_op(mat, little_endian: bool = True)

    将一个基于qubit的矩阵表示转换为对应的泡利算符表示。默认以小端头表示输出QubitOperator。

    参数：
        - **mat** - 基于qubit的矩阵表示。
        - **little_endian** - 是否使用小端头表示(默认为True，即小端头表示)。如果为True，则表示最高位Qubit为最左边的位(即小端头表示)，否则表示最高位Qubit为最右边的位(即大端头表示)

    返回：
        :class:`~.core.QubitOperator`, 对应的泡利算符表示的QubitOperator。