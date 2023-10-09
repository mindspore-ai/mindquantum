mindquantum.io.bprint
======================

.. py:function:: mindquantum.io.bprint(strings: list, align=':', title='', v_around='=', h_around='|', fill_char=' ')

    以block形状打印信息。

    参数：
        - **strings** (list[str]) - 需要输出的信息。
        - **align** (str) - 仅对齐字符的垂直方向，默认值： ``":"``。
        - **title** (str) - 此信息块的标题，默认值： ``""``。
        - **v_around** (str) - 垂直边界字符，默认值： ``"="``。
        - **h_around** (str) - 水平边界字符，默认值： ``"|"``。
        - **fill_char** (str) - 用该字符填充空格。默认值： ``""``。

    返回：
        列表，格式化字符串的列表。
