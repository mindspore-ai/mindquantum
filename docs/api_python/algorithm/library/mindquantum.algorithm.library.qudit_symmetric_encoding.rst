mindquantum.algorithm.library.qudit_symmetric_encoding
========================================================

.. py:function:: mindquantum.algorithm.library.qudit_symmetric_encoding(qudit, n_qudits=1, is_csr=False)

    对称性编码，将qudit态或矩阵编码成qubit对称态或矩阵。

    .. math::

        \begin{align}
        \ket{0}&\to\ket{00\cdots00} \\[.5ex]
        \ket{1}&\to\frac{\ket{0\cdots01}+\ket{0\cdots010}+\ket{10\cdots0}}{\sqrt{d-1}} \\
        \ket{2}&\to\frac{\ket{0\cdots011}+\ket{0\cdots0101}+\ket{110\cdots0}}{\sqrt{d-1}} \\
        \vdots&\qquad\vdots \\[.5ex]
        \ket{d-1}&\to\ket{11\cdots11}
        \end{align}

    参数：
        - **qudit** (np.ndarray) - 需要编码的qudit态或矩阵。
        - **n_qudits** (int) - qudit态或矩阵的量子位个数。默认值：``1``。
        - **is_csr** (bool) - 是否以CSR格式返回矩阵。默认值：False。

    返回：
        np.ndarray，对称性编码后的qubit对称态或矩阵。