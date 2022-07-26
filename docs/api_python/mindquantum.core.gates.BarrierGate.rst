.. py:class:: mindquantum.core.gates.BarrierGate(show=True)

    栅栏门只在量子线路图绘制中产生效果，设置栅栏门后系统不会将栅栏门两边的量子门绘制到同一层。

    参数：
        - **show** (bool) - 是否展示栅栏门。默认值：True.

    .. py:method:: on(obj_qubits, ctrl_qubits=None)

        定义该门作用在哪些量子比特上，并受哪些量子比特控制。

        .. note::
            栅栏门会作用在所有比特上，因此调用该接口总是会产生错误。

        参数：
            - **obj_qubits** (int, list[int]) - 指明量子门作用在哪些量子比特上。
            - **ctrl_qubits** (int, list[int]) - 指明量子门受哪些量子比特控制。默认值：None。

        返回：
            量子门，返回一个新的量子门。
