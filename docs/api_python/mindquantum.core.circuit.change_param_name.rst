.. py:function:: mindquantum.core.circuit.change_param_name(circuit_fn, name_map)

    更改含参量子线路或含参量子算子里的参数名称（是一个可以生成含参量子线路的函数）。

    **参数：**

    - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。
    - **name_map** (dict) - 参数名称映射的dict。

    **异常：**

    - **TypeError** - 如果name_map不是映射。
    - **TypeError** - 如果name_map的key不是字符串。
    - **TypeError** - 如果name_map的value不是字符串。
    - **TypeError** - 如果circuit_fn不是Circuit或不能返回Circuit。

    **返回：**

    Circuit或可以生成Circuit的函数。
