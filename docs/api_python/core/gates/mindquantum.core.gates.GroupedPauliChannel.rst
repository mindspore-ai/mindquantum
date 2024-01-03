mindquantum.core.gates.GroupedPauliChannel
==========================================

.. py:class:: mindquantum.core.gates.GroupedPauliChannel(probs: npt.NDArray[np.float64], **kwargs)

    组合泡利信道。

    该信道等价与一组泡利信道，但是在模拟时，会比一个一个作用泡利信道快很多。关于泡利信道的更多细节，请参考 :class:`~.core.gates.PauliChannel`。

    泡利信道的数学表示如下：

    .. math::

        \epsilon(\rho) = \otimes_i \epsilon_\text{pauli}^i(\rho)

    参数：
        - **probs** (numpy.ndarray) - 所有泡利信道的误差概率。该参数的维度是 `(n, 3)`，其中第一个维度 `n` 表示该组合泡利信道的比特数。第二个维度 `3` 表示每个信道分别发生 :math:`X`， :math:`Y` 或 :math:`Z` 翻转的概率。


    .. py:method:: get_cpp_obj()

        返回量子门的c++对象。

    .. py:method:: matrix()

        返回该噪声信道的Kraus算符。

        返回：
            list，包含了所有泡利信道的Kraus算符。
