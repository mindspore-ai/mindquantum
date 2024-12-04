mindquantum.algorithm.library.qutrit_symmetric_ansatz
=======================================================

.. py:function:: mindquantum.algorithm.library.qutrit_symmetric_ansatz(gate: UnivMathGate, basis: str = "zyz", with_phase: bool = False)

    构造一个保持任意qutrit门编码对称性的qubit ansatz。

    该函数构造一个参数化量子线路(ansatz)来实现任意qutrit门,同时保持qutrit-qubit映射所需的对称性。
    对称性保持意味着在相同对称子空间中的态在门操作后将保持相等的振幅。

    对于单个qutrit(映射到2个qubit),对称子空间为:

    - :math:`\{|00\rangle\}` 对应qutrit态 :math:`|0\rangle`
    - :math:`\{|01\rangle, |10\rangle\}` 对应qutrit态 :math:`|1\rangle`
    - :math:`\{|11\rangle\}` 对应qutrit态 :math:`|2\rangle`

    参考文献：
    `Synthesis of multivalued quantum logic circuits by elementary gates <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.012325>`_，
    `Optimal synthesis of multivalued quantum circuits <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062317>`_。

    参数：
        - **gate** (:class:`~.core.gates.UnivMathGate`) - 由qutrit门编码而来的qubit门。
        - **basis** (str) - 分解的基,可以是 ``"zyz"`` 或者 ``"u3"`` 中的一个。ZYZ基使用RZ和RY旋转,而U3基使用U3门。默认值： ``"zyz"``。
        - **with_phase** (bool) - 是否将全局相位以 :class:`~.core.gates.GlobalPhase` 的形式作用在量子线路上。默认值： ``False``。

    返回：
        :class:`~.core.circuit.Circuit`，保持qutrit编码对称性的qubit ansatz。

    异常：
        - **ValueError** - 如果输入门不是对称的,或者qubit数量与qutrit编码不兼容（必须是2或4个qubit）。