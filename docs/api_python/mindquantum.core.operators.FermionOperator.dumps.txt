mindquantum.core.operators.FermionOperator.dumps(indent=4)

        将FermionOperator转储到JSON（JavaScript对象表示法）

        参数:
            indent (int): 然后JSON数组元素和对象成员将使用该缩进级别漂亮打印。默认值：4。

        返回:
            JSON(str),FermionOperator的JSON字符串

        样例:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> print(f.dumps())
            {
                "((0, 0),)": "(1+2j)",
                "((0, 1),)": "{"a": 1, "__class__": "ParameterResolver", "__module__":                     "parameterresolver.parameterresolver", "no_grad_parameters": []}",
                "__class__": "FermionOperator",
                "__module__": "operators.fermion_operator"
            }
        