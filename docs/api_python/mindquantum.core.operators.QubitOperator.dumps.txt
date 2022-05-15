mindquantum.core.operators.QubitOperator.dumps(indent=4)

        将QubitOperator转储到JSON（JavaScript对象表示法）

        参数:
            indent (int): 然后JSON数组元素和对象成员将使用缩进级别漂亮打印。默认值：4。

        返回:
            JSON（字符串）,QubitOperator的JSON字符串。

        样例:
            >>> from mindquantum.core.operators import QubitOperator
            >>> ops = QubitOperator('X0 Y1', 1.2) + QubitOperator('Z0 X1', {'a': 2.1})
            >>> print(ops.dumps())
            {
                "((0, 'X'), (1, 'Y'))": "1.2",
                "((0, 'Z'), (1, 'X'))": "{"a": 2.1, "__class__": "ParameterResolver",                     "__module__": "parameterresolver.parameterresolver", "no_grad_parameters": []}",
                "__class__": "QubitOperator",
                "__module__": "operators.qubit_operator"
            }
        