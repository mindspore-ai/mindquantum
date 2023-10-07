mindquantum.algorithm.library.bitphaseflip_operator
====================================================

.. py:function:: mindquantum.algorithm.library.bitphaseflip_operator(phase_inversion_index, n_qubits)

    此算子生成一个可以翻转任意计算基的符号的电路。

    参数：
        - **phase_inversion_index** (list[int]) - 需要翻转相位的计算基的索引。
        - **n_qubits** (int) - 量子比特总数。

    返回：
        Circuit，位相位翻转后的线路。
