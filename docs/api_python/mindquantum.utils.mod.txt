mindquantum.utils.mod(vec_in, axis=0)

    计算输入向量的mod。

    参数:
        vec_in (Union[list[numbers.Number], numpy.ndarray]): 要计算mod的向量。
        axis (int): 要沿着哪个轴计算mod。默认值：0。

    返回:
        numpy.nd数组，输入向量的mod。

    样例:
        >>> from mindquantum.utils import mod
        >>> vec_in = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mod(vec_in)
        array([[4.12310563, 5.38516481, 6.70820393]])
        >>> mod(vec_in, 1)
        array([[3.74165739],
               [8.77496439]])
    