mindquantum.utils.SingleLoopProgress
====================================

.. py:class:: mindquantum.utils.SingleLoopProgress(n_loop: int, loop_name: str = 'Task')

    一个用于展示拥有一层循环任务的进度的进度条。

    参数：
        - **n_loop** (int) - 循环的个数。
        - **loop_name** (str) - 循环的进度条的标题。默认值： ``"Task"``。

    .. py:method:: update_loop(loop_idx: int)

        更新循环进度条。

        参数：
            - **loop_idx** (int) - 当前循环的序号。
