mindquantum.algorithm.library.qudit_symmetric_decoding
========================================================

.. py:function:: mindquantum.algorithm.library.qudit_symmetric_decoding(qubit: np.ndarray, n_qubits: int = 1)

    对称性解码，将qubit对称态或矩阵解码成qudit态或矩阵。

    输入的qubit状态或矩阵必须保持qudit-qubit映射所需的对称性。例如，在qutrit(d=3)到双比特的映射中：

    .. math::

        \begin{align}
        \ket{00\cdots00}&\to\ket{0} \\[.5ex]
        \frac{\ket{0\cdots01}+\ket{0\cdots010}+\ket{10\cdots0}}{\sqrt{d-1}}&\to\ket{1} \\
        \frac{\ket{0\cdots011}+\ket{0\cdots0101}+\ket{110\cdots0}}{\sqrt{d-1}}&\to\ket{2} \\
        \vdots&\qquad\vdots \\[.5ex]
        \ket{11\cdots11}&\to\ket{d-1}
        \end{align}

    这种对称性要求在同一对称子空间内的状态必须具有相等的振幅。例如，状态 :math:`|01\rangle` 和 :math:`|10\rangle` 属于同一对称子空间，必须具有相等的振幅。

    参数：
        - **qubit** (np.ndarray) - 需要解码的qubit对称态或矩阵，qubit态或矩阵需满足对称性。
        - **n_qubits** (int) - qubit对称态或矩阵的量子比特数。默认值：``1``。

    返回：
        np.ndarray，对称性解码后的qudit态或矩阵。

    异常：
        - **ValueError** - 如果输入的qubit状态或矩阵不保持所需的对称性。
