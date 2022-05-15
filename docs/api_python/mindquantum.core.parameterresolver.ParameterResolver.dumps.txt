mindquantum.core.parameterresolver.ParameterResolver.dumps(indent=4)

        将参数解析器转储到JSON（JavaScript对象表示法）

        参数：
            indent (int)：然后JSON数组元素和对象成员将使用该缩进级别，打印完整。默认值：4。

        返回：
            string(JSON)，参数解析器的JSON

        样例：
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2, 'c': 3, 'd': 4})
            >>> pr.no_grad_part('a', 'b')
            >>> print(pr.dumps())
            {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "__class__": "ParameterResolver",
                "__module__": "parameterresolver",
                "no_grad_parameters": [
                    "a",
                    "b"
                ]
            }
 }
