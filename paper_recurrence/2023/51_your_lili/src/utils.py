"""Some utilities functions."""

from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit


def c2q(qlist: list, value: int) -> Circuit:
    """将经典计算机数字转换成量子线路
    Args:
        qlist:  list[int]: 作用的qubit，低位在后，如 [3,2,1,0]
        value:  int:       十进制的加数或减数，非负
    Return:
        准备了初态的线路
    """
    vbin = bin(value)[2:][::-1]   # 将十进制转换成二进制，并且低位在前
    qlist = qlist[::-1]           # 低位在前
    cir = Circuit([X.on(q) for q, v in zip(qlist, vbin) if v == '1'])
    return cir


def q2c(cir: Circuit, qlist: list) -> int:
    """获取量子线路状态，并得到对应的经典数值
    Args:
        cir: 量子线路
        qlist: 保存结果的量子位，对 [q2, q1, q0] 读取到的数为 int(q2q1q0)
    Return:
        十进制的量子态表示的数值
    """
    ket_str = cir.get_qs(ket=True)   # 获取字符串表示的状态
    print(f'ket string: {ket_str}')
    # 解析字符串得到十进制的数值
    res_str = ""
    for q in qlist:
        # |0101>，需要逆序和除去最后的ket符号
        res_str += ket_str[-q-2]
    return int(res_str, base=2)
