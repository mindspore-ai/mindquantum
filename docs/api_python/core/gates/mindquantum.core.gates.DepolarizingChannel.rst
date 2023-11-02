mindquantum.core.gates.DepolarizingChannel
===========================================

.. py:class:: mindquantum.core.gates.DepolarizingChannel(p: float, n_qubits: int = 1, **kwargs)

    去极化信道。描述的噪声体现为：以 :math:`P` 的概率将量子态转变为最大混态（随机作用泡利门（I、X、Y、Z）的其中一个，每个泡利门的概率都是 :math:`P/4` ），或以 :math:`1-P` 的概率保持不变。

    对于单比特情况，去极化信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P/4( I \rho I + X \rho X + Y \rho Y + Z \rho Z)

    其中，:math:`\rho` 是密度矩阵形式的量子态；:math:`P` 是发生去极化错误的概率。

    该信道还支持作用于多个目标比特。在 :math:`N` 比特情况下，去极化信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + \frac{P}{4^N} \sum_j U_j \rho U_j

    其中，:math:`N` 是目标比特数； :math:`U_j \in \left\{ I, X, Y, Z \right\} ^{\otimes N}` 多比特泡利算符。

    * 当 :math:`0 \le P \le 1` 时, 该信道是去极化信道, 并且当 :math:`P = 1` 时是完全去极化信道。
    * 然而，:math:`1 < P \le 4^N / (4^N - 1)`同样是合法的情况, 但此时不再是去极化信道。当 :math:`P = 4^N / (4^N - 1)`
      时它变为均匀泡利信道：:math:`E(\rho) = \sum_j V_j \rho V_j / (4^n - 1)`，其中 :math:`V_j = U_j \setminus I^{\otimes N}`。

    参数：
        - **p** (int, float) - 发生去极化错误的概率。
        - **n_qubits** (int) - 去极化信道的比特数。默认值：``1``。

    .. py:method:: get_cpp_obj()

        返回量子门的c++对象。

    .. py:method:: matrix()

        返回该噪声信道的Kraus算符。

        返回：
            list，包含了该噪声信道的Kraus算符，且顺序为 :math:`\left\{ I, X, Y, Z \right\} ^{\otimes N}` 的字典顺序。
