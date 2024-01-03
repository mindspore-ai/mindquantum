mindquantum.core.parameterresolver.PRGenerator
==============================================

.. py:class:: mindquantum.core.parameterresolver.PRGenerator(name='p', prefix: str = '', suffix: str = '', dtype=None)

    一个一个的生成参数。

    参数：
        - **name** (str) - 变量的主要名称。默认值： ``'p'``。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
        - **dtype** (mindquantum.dtype) - 改参数解析器的数据类型。如果为 ``None``，则类型为 ``mindquantum.float64``。默认： ``None``。

    .. py:method:: new(prefix: str = '', suffix: str = '')

        生成下一个新的参数。

        参数：
            - **prefix** (str) - 生成此参数时的额外前缀。默认值： ``''``。
            - **suffix** (str) - 生成此参数时的额外后缀。默认值： ``''``。

    .. py:method:: reset()

        重置参数生成器到初态。

    .. py:method:: size()

        返回已生成的参数的个数。
