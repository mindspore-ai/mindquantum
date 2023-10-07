mindquantum.core.gates.PauliChannel
====================================

.. py:class:: mindquantum.core.gates.PauliChannel(px: float, py: float, pz: float, **kwargs)

    泡利信道。描述的噪声体现为：在量子比特上随机作用一个额外的泡利门，作用 :math:`X` 、 :math:`Y` 和 :math:`Z` 门对应概率分别为 :math:`P_x` 、 :math:`P_y` 和 :math:`P_z` ，或以概率 :math:`1-P_x-P_y-P_z` 的概率保持不变（作用 :math:`I` 门）。

    泡利信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P_x - P_y - P_z)\rho + P_x X \rho X + P_y Y \rho Y + P_z Z \rho Z

    其中， :math:`\rho` 是密度矩阵形式的量子态； :math:`P_x` 、 :math:`P_y` 和 :math:`P_z` 是作用的泡利门为 :math:`X` 、 :math:`Y` 和 :math:`Z` 门的概率。

    参数：
        - **px** (int, float) - 作用的泡利门是X门的概率。
        - **py** (int, float) - 作用的泡利门是Y门的概率。
        - **pz** (int, float) - 作用的泡利门是Z门的概率。

    .. py:method:: define_projectq_gate()

        定义对应的projectq门。

    .. py:method:: get_cpp_obj()

        返回量子门的c++对象。

    .. py:method:: matrix()

        返回该噪声信道的Kraus算符。

        返回：
            list，包含了该噪声信道的Kraus算符。
