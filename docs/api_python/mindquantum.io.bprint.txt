mindquantum.io.bprint(strings: list, align=':', title='', v_around='=', h_around='|', fill_char=' ')

    以块状打印信息。

    参数:
        strings (list[str]): 要输出的信息。
        align (str): 仅对齐字符垂直方向。默认值：“：”。
        title (str): 此信息块的标题。默认值：“”。
        v_around (str): 垂直边界字符。默认值："="。
        h_around (str): 水平边界字符。默认值：“|”。
        fill_char (str): 用这个字符填充空格。默认值：“”。

    返回:
        list，格式化字符串的列表。

    样例:
        >>> from mindquantum.io import bprint
        >>> title='Info of Bob'
        >>> o = bprint(['Name:Bob', 'Age:17', 'Nationality:China'],
        ...     title=title)
        >>> for i in o:
        ...     print(i)
        ====Info of Bob====
        |Name       :Bob  |
        |Age        :17   |
        |Nationality:China|
        ===================
    