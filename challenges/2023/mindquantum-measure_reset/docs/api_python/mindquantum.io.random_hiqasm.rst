mindquantum.io.random_hiqasm
=============================

.. py:function:: mindquantum.io.random_hiqasm(n_qubits, gate_num, version='0.1', seed=42)

    生成随机的hiqasm格式支持的量子线路。

    参数：
        - **n_qubits** (int) - 此量子线路中的量子比特总数。
        - **gate_num** (int) - 此量子线路中的门总数。
        - **version** (str) - HIQASM的版本。默认值： ``"0.1"``。
        - **seed** (int) - 生成此随机量子线路的随机种子。默认值： ``42``。

    返回：
        str，HIQASM格式的量子线路。
