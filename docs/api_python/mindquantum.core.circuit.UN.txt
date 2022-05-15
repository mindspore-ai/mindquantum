Class mindquantum.core.circuit.UN(gate: mindquantum.core.gates.basic.BasicGate, maps_obj, maps_ctrl=None)

    将量子门映射到不同的目标量子位和控制量子位。

    参数:
        gate (BasicGate): 量子门。
        maps_obj (Union[int, list[int]]): 目标量子位。
        maps_ctrl (Union[int, list[int]]): 控制量子位。默认值：None。

    返回:
        电路，返回量子电路。

    样例:
        >>> from mindquantum.core import UN, X
        >>> circuit1 = UN(X, maps_obj = [0, 1], maps_ctrl = [2, 3])
        >>> print(circuit1)
        q0: ──X───────
              │
        q1: ──┼────X──
              │    │
        q2: ──●────┼──
                   │
        q3: ───────●──
        >>> from mindquantum.core import SWAP
        >>> circuit2 = UN(SWAP, maps_obj =[[0, 1], [2, 3]]).x(2, 1)
        >>> print(circuit2)
        q0: ──@───────
              │
        q1: ──@────●──
                   │
        q2: ──@────X──
              │
        q3: ──@───────
       