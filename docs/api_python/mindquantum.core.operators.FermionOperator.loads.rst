mindquantum.core.operators.FermionOperator.loads(strs)

        将JSON（JavaScript对象表示法）加载到FermionOperator中

        参数:
            strs (str): 转储的费米子运算符字符串。

        返回:
            FermionOperator，从字符串加载的FermionOperator

        样例:
            >>> from mindquantum.core.operators import FermionOperator
            >>> strings == '{"((0, 0),)": "(1+2j)", "((0, 1),)": {"a": 1},                 "__class__": "FermionOperator", "__module__": "__main__"}'
            >>> obj = FermionOperator.loads(strings)
            >>> print(obj)
            (1+2j) [0] + a [0^]
        