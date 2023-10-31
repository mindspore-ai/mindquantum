mindquantum.utils.TwoLoopsProgress
==================================

.. py:class:: mindquantum.utils.TwoLoopsProgress(n_outer_loop: int, n_inner_loop: int, outer_loop_name: str = 'Epoch', inner_loop_name: str = 'Batch')

    一个用于展示拥有两层循环任务的进度的进度条。

    参数：
        - **n_outer_loop** (int) - 外层循环的个数。
        - **n_inner_loop** (int) - 内层循环的个数。
        - **outer_loop_name** (str) - 外层循环的进度条的标题。默认值： ``"Epoch"``。
        - **inner_loop_name** (str) - 内层循环的进度条的标题。默认值： ``"Batch"``。

    .. py:method:: update_inner_loop(loop_idx: int)

        更新内层循环进度条。

        参数：
            - **loop_idx** (int) - 当前内层循环的序号。

    .. py:method:: update_outer_loop(loop_idx: int)

        更新外层循环进度条。

        参数：
            - **loop_idx** (int) - 当前外层循环的序号。
