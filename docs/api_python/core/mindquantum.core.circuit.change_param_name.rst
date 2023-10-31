mindquantum.core.circuit.change_param_name
===========================================

.. py:function:: mindquantum.core.circuit.change_param_name(circuit_fn, name_map)

    更改含参量子线路或含参量子算子（可以生成含参量子线路的函数）里的参数名称。

    参数：
        - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。
        - **name_map** (dict) - 参数名称映射的dict。

    返回：
        Circuit，或可以生成Circuit的函数。

    异常：
        - **TypeError** - 如果 `name_map` 不是映射。
        - **TypeError** - 如果 `name_map` 的 `key` 不是字符串。
        - **TypeError** - 如果 `name_map` 的 `value` 不是字符串。
        - **TypeError** - 如果 `circuit_fn` 不是Circuit或不能返回Circuit。
