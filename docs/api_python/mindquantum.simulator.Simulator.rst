.. py:class:: mindquantum.simulator.Simulator(backend, n_qubits, seed=None)

    模拟量子电路的量子模拟器。

    **参数：**

    - **backend** (str) - 想要的后端。通过调用 `get_supported_simulator()` 可以返回支持的后端。
    - **n_qubits** (int) - 量子模拟器的数量。
    - **seed** (int) - 模拟器的随机种子，如果为None，种子将由 `numpy.random.randint` 生成。默认值：None。

    **异常：**

    - **TypeError** - 如果 `backend` 不是str。
    - **TypeError** - 如果 `n_qubits` 不是int。
    - **TypeError** - 如果 `seed` 不是int。
    - **ValueError** - 如果不支持 `backend` 。
    - **ValueError** - 如果 `n_qubits` 为负数。
    - **ValueError** - 如果 `seed` 小于0或大于2**23 - 1。