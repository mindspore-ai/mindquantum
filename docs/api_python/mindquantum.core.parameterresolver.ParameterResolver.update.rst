.. py:method:: mindquantum.core.parameterresolver.ParameterResolver.update(other)

    使用其它参数解析器更新此参数解析器。

    **参数：**

    - **others** (ParameterResolver) - 其它参数解析器。

    **异常：**

    - **ValueError** – 如果某些参数需要grad而在其它参数解析器中不需要grad，反之亦然，某些参数是编码器参数而在其它参数解析器中不是编码器。
