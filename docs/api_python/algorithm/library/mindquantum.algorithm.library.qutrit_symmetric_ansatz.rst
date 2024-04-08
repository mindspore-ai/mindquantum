mindquantum.algorithm.library.qutrit_symmetric_ansatz
=======================================================

.. py:function:: qutrit_symmetric_ansatz(gate: UnivMathGate, basis: str = "zyz", with_phase: bool = False)

    构造一个保持任意qutrit门编码对称性的qubit ansatz。

    参考文献：
    `Synthesis of multivalued quantum logic circuits by elementary gates <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.012325>`_，
    `Optimal synthesis of multivalued quantum circuits <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062317>`_。

    参数：
        - **gate** (:class:`~.core.gates.UnivMathGate`) - 由qutrit门编码而来的qubit门。
        - **basis** (str) - 分解的基，可以是 ``"zyz"`` 或者 ``"u3"`` 中的一个。默认值： ``"zyz"``。
        - **with_phase** (bool) - 是否将全局相位以 :class:`~.core.gates.GlobalPhase` 的形式作用在量子线路上。默认值： ``False``。

    返回：
        :class:`~.core.circuit.Circuit`，保持qutrit编码对称性的qubit ansatz。