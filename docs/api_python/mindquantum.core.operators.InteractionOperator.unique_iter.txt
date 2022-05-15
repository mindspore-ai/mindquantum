mindquantum.core.operators.InteractionOperator.unique_iter(complex_valued=False)

        迭代不在同一对称组中的所有项。

        四点对称：
            1. pq = qp.
            2. pqrs = srqp = qpsr = rspq.
        八点对称（当复杂值为假时）：
            1. pq = qp.
            2. pqrs = rqps = psrq = srqp = qpsr = rspq = spqr = qrsp.

        参数:
            complex_valued (bool): 运算符是否具有复杂系数。默认值：False。
        