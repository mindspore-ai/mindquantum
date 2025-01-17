mindquantum.algorithm.nisq.SGAnsatz2D
========================================

.. py:class:: mindquantum.algorithm.nisq.SGAnsatz2D(nqubits, k, line_set=None, nlayers=1, prefix='', suffix='')

    序列生成（SG）ansatz，用于二维量子系统。

    SG ansatz由多个变分量子电路块组成，每个块都是应用于相邻量子比特的参数化量子电路。这种结构使得SG ansatz天然适合于量子多体问题。

    具体而言，对于一维量子系统，SG ansatz可以高效地生成具有固定键维度的任意矩阵乘积态。对于二维系统，SG ansatz可以生成 string-bond 态。

    了解更多详细信息，请访问 `A sequentially generated variational quantum circuit with polynomial complexity <https://arxiv.org/abs/2305.12856>`_。

    参数：
        - **nqubits** (int) - ansatz中的量子比特数。
        - **k** (int) - log(R) + 1，其中R是固定的键维度。
        - **line_set** (list, optional) - 量子比特线路集合，用于生成特定类型的字符串键态。如果为None，将自动生成为1×N的网格，其中N等于nqubits。默认值：``None``。
        - **nlayers** (int) - 每个块中的层数。默认值：``1``。
        - **prefix** (str) - 参数的前缀。默认值：``''``。
        - **suffix** (str) - 参数的后缀。默认值：``''``。

    .. py:method:: from_grid(nrow, ncol, k, nlayers=1, prefix='', suffix='')
        :classmethod:

        从二维网格配置创建SGAnsatz2D。

        这是为二维量子系统创建SGAnsatz2D实例的推荐方法。它会根据网格维度自动生成适当的线路集合。

        参数：
            - **nrow** (int) - 二维网格中的行数。
            - **ncol** (int) - 二维网格中的列数。
            - **k** (int) - log(R) + 1，其中R是固定的键维度。
            - **nlayers** (int) - 每个块中的层数。默认值：``1``。
            - **prefix** (str) - 参数的前缀。默认值：``''``。
            - **suffix** (str) - 参数的后缀。默认值：``''``。

        返回：
            SGAnsatz2D，为指定二维网格配置的新实例。

    .. py:method:: generate_line_set(nrow, ncol)
        :classmethod:

        为二维量子系统生成蛇形遍历模式。

        此方法为二维量子系统生成两种不同的遍历路径：
        1. 列向蛇形模式：在上下方向交替遍历每一列
        2. 行向蛇形模式：在左右方向交替遍历每一行

        参数：
            - **nrow** (int) - 二维网格中的行数。
            - **ncol** (int) - 二维网格中的列数。

        返回：
            list，包含两个遍历路径的列表，每个路径是一个量子比特索引列表。第一个路径是列向的，第二个是行向的。
