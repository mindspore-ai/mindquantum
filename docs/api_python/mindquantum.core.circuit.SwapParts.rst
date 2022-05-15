Class mindquantum.core.circuit.SwapParts(a: collections.abc.Iterable, b: collections.abc.Iterable, maps_ctrl=None)

    交换量子电路的两个不同部分，有或没有控制量子位。

    参数:
        a (Iterable): 您需要交换的第一部分。
        b (Iterable): 您需要交换的第二部分。
        maps_ctrl (int, Iterable): 通过单个量子位或不同量子位控制交换，或者只是不控制量子位。默认值：None。

    样例:
        >>> from mindquantum import SwapParts
        >>> SwapParts([1, 2], [3, 4], 0)
        q0: ──●────●──
              │    │
        q1: ──@────┼──
              │    │
        q2: ──┼────@──
              │    │
        q3: ──@────┼──
                   │
        q4: ───────@──
       