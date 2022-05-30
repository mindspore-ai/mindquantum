.. py:class:: mindquantum.algorithm.nisq.IQPEncoding(n_feature, first_rotation_gate=<class 'mindquantum.core.gates.basicgate.RZ'>, second_rotation_gate=<class 'mindquantum.core.gates.basicgate.RZ'>, num_repeats=1)

    通用IQP编码。

    **参数：**

    - **n_feature** (int) – IQPEncoding的数据特征数。
    - **first_rotation_gate** (ParamaterGate) – 旋转门RX、RY或RZ之一。
    - **second_rotation_gate** (ParamaterGate) – 旋转门RX、RY或RZ之一。
    - **num_repeats** (int) – 编码迭代次数。
       