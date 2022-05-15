mindquantum.simulator.inner_product(bra_simulator: mindquantum.simulator.simulator.Simulator, ket_simulator: mindquantum.simulator.simulator.Simulator)

    计算给定模拟器中两种状态的内积。

    参数:
        bra_simulator (Simulator): 用作布拉状态的模拟器。
        ket_simulator (Simulator): 用作ket状态的模拟器。

    返回:
        数字。数字，两个量子态的内积。

    样例:
        >>> from mindquantum import RX, RY, Simulator
        >>> from mindquantum.simulator import inner_product
        >>> bra_simulator = Simulator('projectq', 1)
        >>> bra_simulator.apply_gate(RY(1.2).on(0))
        >>> ket_simulator = Simulator('projectq', 1)
        >>> ket_simulator.apply_gate(RX(2.3).on(0))
        >>> inner_product(bra_simulator, ket_simulator)
    