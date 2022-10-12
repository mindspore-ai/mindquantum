src.ansatz\_mpo
==================

量子线路生成。

|

.. py:function:: src.ansatz_mpo.generate_ansatz_mpo(n, L, barrier=False)

    用于生成MPO(matrix product operators)的量子线路。

    参数：
        - **n** (int) - 量子比特数，需大于2。
        - **L** (int) - 量子线路层数。
        - **barrier** (bool) - 是否添加barrier，默认False。

    返回：
        Circuit，能够作为MPO的量子线路。
