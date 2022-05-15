mindquantum.core.operators.QubitOperator.loads(strs)

        将JSON（JavaScript对象表示法）加载到QubitOperator中。

        参数:
            strs (str): 转储的量子位运算符字符串。

        返回:
            FermionOperator，从字符串加载的QubitOperator

        样例:
            >>> from mindquantum.core.operators import QubitOperator
            >>> strings = """{"((0, 'X'), (1, 'Y'))": 1.2, "((0, 'Z'), (1, 'X'))": {"a": 2.1},                 "__class__": "QubitOperator", "__module__": "__main__"}"""
            >>> obj = QubitOperator.loads(strings)
            >>> print(obj)
            1.2 [X0 Y1] + 2.1*a [Z0 X1]
        