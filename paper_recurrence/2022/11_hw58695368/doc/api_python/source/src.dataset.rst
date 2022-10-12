src.dataset
==============

数据处理、评价。

|

.. py:function:: src.dataset.score(problem, res)

    切割情况 :math:`\mathcal{C}` 对图 :math:`G` 的评分。

    .. math::

            Score = \displaystyle\sum_{i,j}^{n} \frac{G_{ij}}{2}(1-\mathcal{C}_i*\mathcal{C}_j)


    参数：
        - **problem** (Union[numpy.ndarray, list]) - 图。
        - **res** (Union[numpy.ndarray, list]) - 切割情况。

    返回：
        float，得分。

|

.. py:function:: src.dataset.build_dataset_parallel(n1, problem1, n2, problem2)

    构建MBE最大切割求解器并行求解所需的新图。

    参数：
        - **n1** (int) - 图1顶点数。
        - **problem1** (Union[numpy.ndarray, list]) - 待求解图1。
        - **n2** (int) - 图2顶点数。
        - **problem** (Union[numpy.ndarray, list]) - 待求解图2。

    返回：
        int，顶点数。
        list，新图。
        list，新图中的原图排列顺序。

|

.. py:function:: src.dataset.build_dataset1()

    测试数据集1。8顶点图。

    返回：
        int，顶点数。
        list，图。

|

.. py:function:: src.dataset.build_dataset2()

    测试数据集2。8顶点二分图。

    返回：
        int，顶点数。
        list，图。

|

.. py:function:: src.dataset.build_dataset3()

    测试数据集3。10顶点3正则图。

    返回：
        int，顶点数。
        list，图。
