mindquantum.utils.normalize(vec_in, axis=0)

    根据指定的轴归一化输入向量。

    参数:
        vec_in (Union[list[number], numpy.ndarray]): 要标准化的向量。
        axis (int): 要沿着哪个轴标准化矢量。默认值：0。

    返回:
        numpy.nd数组，归一化后的矢量。

    样例:
        >>> from mindquantum.utils import normalize
        >>> vec_in = np.array([[1, 2, 3], [4, 5, 6]])
        >>> normalize(vec_in)
        array([[0.24253563, 0.37139068, 0.4472136 ],
               [0.9701425 , 0.92847669, 0.89442719]])
        >>> normalize(vec_in, 1)
        array([[0.26726124, 0.53452248, 0.80178373],
               [0.45584231, 0.56980288, 0.68376346]])
    