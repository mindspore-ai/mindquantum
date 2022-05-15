mindquantum.utils.ket_string(state, tol=1e-07)

    获取量子状态的ket格式。

    参数:
        state (numpy.ndarray): 输入量子状态。
        tol (float): 忽略小振幅的tol。默认值：1e-7。

    返回:
        str，量子态的ket格式。

    样例:
        >>> import numpy as np
        >>> from mindquantum.utils import ket_string
        >>> state = np.array([1, -1j])/np.sqrt(2)
        >>> print(ket_string(state))
        ['√2/2¦0⟩', '-√2/2j¦1⟩']
       