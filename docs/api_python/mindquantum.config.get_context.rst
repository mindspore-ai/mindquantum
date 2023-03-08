mindquantum.config.get_context
==============================

.. py:function:: mindquantum.config.get_context(attr_key: str)

    根据输入的键获取上下文属性值。

    如果有些属性没有设置，会自动获取。

    参数：
        - **attr_key** (str) - 属性的键。

    返回：
        Object，给定属性键的值。

    异常：
        - **ValueError** - 如果输入键不是上下文中的属性。
