mindquantum.utils.fdopen
=========================

.. py:function:: mindquantum.utils.fdopen(fname, mode, perms=0o600, encoding=None)

    以正确权限打开文件的上下文管理器。

    参数：
        - **fname** (str) - 需要读写的文件的路径。
        - **mode** (str) - 以何种方式打开文件（查询内置函数 `open()` 进行更多帮助）。
        - **perms** (int) - 权限掩码（查询 `os.open()` 进行更多帮助）。
        - **encoding** (str) - 对文件进行编码或者解码的编码器。默认值： ``None``。
