"""The arithmetic circuits."""

from mindquantum.core.gates import X, SWAP
from mindquantum.core.circuit import Circuit, shift, UN


def Adder() -> Circuit:
    """4-bit 加法器，对应 Fig.6
    """
    return Circuit([
        X(10, [4, 0]),
        X(11, [5, 1]),
        X(12, [6, 2]),
        X(8, [7, 3]),
        X(4, 0),
        X(5, 1),
        X(6, 2),
        X(7, 3),
        X(10, [9, 4]),
        X(11, [10, 5]),
        X(12, [11, 6]),
        X(8, [7, 12]),
        X(7, 12),
        X(12, [11, 6]),
        X(6, 2),
        X(12, [6, 2]),
        X(6, 2),
        X(6, 11),
        X(11, [10, 5]),
        X(5, 1),
        X(11, [5, 1]),
        X(5, 1),
        X(5, 10),
        X(10, [9, 4]),
        X(4, 0),
        X(10, [4, 0]),
        X(4, 0),
        X(4, 9)
    ])


def AdderRev() -> Circuit:
    """4-bit 逆加法器（减法器），对应 Fig.7
    """
    return Circuit([
        X(4, 9),
        X(4, 0),
        X(10, [4, 0]),
        X(4, 0),
        X(10, [9, 4]),
        X(5, 10),
        X(5, 1),
        X(11, [5, 1]),
        X(5, 1),
        X(11, [10, 5]),
        X(6, 11),
        X(6, 2),
        X(12, [6, 2]),
        X(6, 2),
        X(12, [11, 6]),
        X(7, 12),
        X(8, [7, 12]),
        X(7, 3),
        X(12, [6, 2]),
        X(8, [7, 3]),
        X(6, 2),
        X(12, [11, 6]),
        X(11, [5, 1]),
        X(5, 1),
        X(11, [10, 5]),
        X(10, [4, 0]),
        X(4, 0),
        X(10, [9, 4])
    ])


def AdderRev2() -> Circuit:
    """逆-加法器"""
    cir = Adder()
    cir.reverse()
    return cir


def ModAdder() -> Circuit:
    """模加法器，对应 Fig.8
    """
    cir = Circuit()
    cir += Adder()
    cir += UN(SWAP, maps_obj=[(0, 13), (1, 14), (2, 15), (3, 16)])
    cir += AdderRev2()
    cir += Circuit([X(8), X(17, 8), X(8)])
    cir += UN(X, maps_obj=[0, 1, 2, 3], maps_ctrl=[17, 17, 17, 17])
    cir += Adder()
    cir += UN(X, maps_obj=[0, 1, 2, 3], maps_ctrl=[17, 17, 17, 17])
    cir += UN(SWAP, maps_obj=[(0, 13), (1, 14), (2, 15), (3, 16)])
    cir += AdderRev2()
    cir += X(17, 8)
    cir += Adder()
    return cir


def ModAdderRev():
    """逆-模加法器, 对应Fig.9
    """
    cir = Circuit()
    cir += AdderRev2()
    cir += X(17, 8)
    cir += Adder()
    cir += UN(SWAP, maps_obj=[(0, 13), (1, 14), (2, 15), (3, 16)])
    cir += UN(X, maps_obj=[0, 1, 2, 3], maps_ctrl=[17, 17, 17, 17])
    cir += AdderRev2()
    cir += UN(X, maps_obj=[0, 1, 2, 3], maps_ctrl=[17, 17, 17, 17])
    cir += Circuit([X(8), X(17, 8), X(8)])
    cir += Adder()
    cir += UN(SWAP, maps_obj=[(0, 13), (1, 14), (2, 15), (3, 16)])
    cir += AdderRev2()
    return cir


def ModMulti7xmod15():
    """4-bit 模乘法器，求 7x mod 15. 对应 Fig.10
    """
    ns = 5  # 模加法器所需 shift 量
    cir = Circuit()
    cir += UN(X, maps_obj=[5, 6, 7], maps_ctrl=[[0, 1], [0, 1], [0, 1]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 6, 7], maps_ctrl=[[0, 1], [0, 1], [0, 1]])

    cir += UN(X, maps_obj=[6, 7, 8], maps_ctrl=[[0, 2], [0, 2], [0, 2]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[6, 7, 8], maps_ctrl=[[0, 2], [0, 2], [0, 2]])

    cir += UN(X, maps_obj=[5, 7, 8], maps_ctrl=[[0, 3], [0, 3], [0, 3]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 7, 8], maps_ctrl=[[0, 3], [0, 3], [0, 3]])

    cir += UN(X, maps_obj=[5, 6, 8], maps_ctrl=[[0, 4], [0, 4], [0, 4]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 6, 8], maps_ctrl=[[0, 4], [0, 4], [0, 4]])

    cir += X(0)
    cir += UN(X, maps_obj=[9, 10, 11, 12],
              maps_ctrl=[[0, 1], [0, 2], [0, 3], [0, 4]])
    cir += X(0)
    return cir


def ModMulti7xmod15Rev():
    """4-bit 逆-模乘法器，对应 Fig.11
    """
    ns = 5  # 模加法器所需 shift 量
    cir = Circuit()
    cir += X(0)
    cir += UN(X, maps_obj=[12, 11, 10, 9],
              maps_ctrl=[[0, 4], [0, 3], [0, 2], [0, 1]])
    cir += X(0)

    cir += UN(X, maps_obj=[6, 7, 8], maps_ctrl=[[0, 4], [0, 4], [0, 4]])
    cir += shift(ModAdderRev(), ns)
    cir += UN(X, maps_obj=[6, 7, 8], maps_ctrl=[[0, 4], [0, 4], [0, 4]])

    cir += UN(X, maps_obj=[5, 6, 7], maps_ctrl=[[0, 3], [0, 3], [0, 3]])
    cir += shift(ModAdderRev(), ns)
    cir += UN(X, maps_obj=[5, 6, 7], maps_ctrl=[[0, 3], [0, 3], [0, 3]])

    cir += UN(X, maps_obj=[5, 6, 8], maps_ctrl=[[0, 2], [0, 2], [0, 2]])
    cir += shift(ModAdderRev(), ns)
    cir += UN(X, maps_obj=[5, 6, 8], maps_ctrl=[[0, 2], [0, 2], [0, 2]])

    cir += UN(X, maps_obj=[5, 7, 8], maps_ctrl=[[0, 1], [0, 1], [0, 1]])
    cir += shift(ModAdderRev(), ns)
    cir += UN(X, maps_obj=[5, 7, 8], maps_ctrl=[[0, 1], [0, 1], [0, 1]])
    return cir


def ModMulti4xmod15():
    """4-bit 模乘法器，求 4x mod 15.
    """
    ns = 5  # 模加法器所需 shift 量
    cir = Circuit()
    # ModAdd() 两边为 a2^i, a=4此处
    cir += UN(X, maps_obj=[7], maps_ctrl=[[0, 1]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[7, 8], maps_ctrl=[[0, 1], [0, 2]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[8, 5], maps_ctrl=[[0, 2], [0, 3]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 6], maps_ctrl=[[0, 3], [0, 4]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[6], maps_ctrl=[[0, 4]])
    cir += X(0)
    cir += UN(X, maps_obj=[9, 10, 11, 12],
              maps_ctrl=[[0, 1], [0, 2], [0, 3], [0, 4]])
    cir += X(0)
    return cir


def ModMulti1xmod15(ns=5):
    """4-bit 模乘法器，求 1x mod 15."""
    ns = 5  # 模加法器所需 shift 量
    cir = Circuit()
    cir += UN(X, maps_obj=[5], maps_ctrl=[[0, 1]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 6], maps_ctrl=[[0, 1], [0, 2]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[6, 7], maps_ctrl=[[0, 2], [0, 3]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[7, 8], maps_ctrl=[[0, 3], [0, 4]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[8], maps_ctrl=[[0, 4]])
    cir += X(0)
    cir += UN(X, maps_obj=[9, 10, 11, 12],
              maps_ctrl=[[0, 1], [0, 2], [0, 3], [0, 4]])
    cir += X(0)
    return cir


def ModMulti13xmod15():
    """4-bit 模乘法器，求 13x mod 15.
    """
    ns = 5  # 模加法器所需 shift 量
    cir = Circuit()
    cir += UN(X, maps_obj=[5, 7, 8], maps_ctrl=[[0, 1], [0, 1], [0, 1]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 7, 8], maps_ctrl=[[0, 1], [0, 1], [0, 1]])

    cir += UN(X, maps_obj=[5, 6, 8], maps_ctrl=[[0, 2], [0, 2], [0, 2]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 6, 8], maps_ctrl=[[0, 2], [0, 2], [0, 2]])

    cir += UN(X, maps_obj=[5, 6, 7], maps_ctrl=[[0, 3], [0, 3], [0, 3]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[5, 6, 7], maps_ctrl=[[0, 3], [0, 3], [0, 3]])

    cir += UN(X, maps_obj=[6, 7, 8], maps_ctrl=[[0, 4], [0, 4], [0, 4]])
    cir += shift(ModAdder(), ns)
    cir += UN(X, maps_obj=[6, 7, 8], maps_ctrl=[[0, 4], [0, 4], [0, 4]])

    cir += UN(X, maps_obj=[8], maps_ctrl=[[0, 4]])
    cir += X(0)
    cir += UN(X, maps_obj=[9, 10, 11, 12],
              maps_ctrl=[[0, 1], [0, 2], [0, 3], [0, 4]])
    cir += X(0)
    return cir


def ModMulti4xmod15Rev():
    """将 4x mod 15 线路翻转
    """
    cir = ModMulti4xmod15()
    cir.reverse()
    return cir


def ModMulti1xmod15Rev():
    """将 1x mod 15 线路翻转
    """
    cir = ModMulti1xmod15()
    cir.reverse()
    return cir


def ModMulti13xmod15Rev():
    """将 13x mod 15 线路翻转
    """
    cir = ModMulti13xmod15()
    cir.reverse()
    return cir


def ModExp():
    """4-bit 模指数线路，对应 Fig.12
    """
    ns = 4
    cir = Circuit()
    cir += X(4, 0)
    cir += X(5)
    cir += shift(ModMulti7xmod15(), ns)
    cir += UN(SWAP, maps_obj=[(5, 13), (6, 14), (7, 15), (8, 16)])
    cir += shift(ModMulti13xmod15Rev(), ns)

    cir += UN(X, maps_obj=[4, 4], maps_ctrl=[0, 1])
    cir += shift(ModMulti4xmod15(), ns)
    cir += UN(SWAP, maps_obj=[(5, 13), (6, 14), (7, 15), (8, 16)])
    cir += shift(ModMulti4xmod15Rev(), ns)

    cir += UN(X, maps_obj=[4, 4], maps_ctrl=[1, 2])
    cir += shift(ModMulti1xmod15(), ns)
    cir += UN(SWAP, maps_obj=[(5, 13), (6, 14), (7, 15), (8, 16)])
    cir += shift(ModMulti1xmod15Rev(), ns)

    cir += UN(X, maps_obj=[4, 4], maps_ctrl=[2, 3])
    cir += shift(ModMulti1xmod15(), ns)
    cir += UN(SWAP, maps_obj=[(5, 13), (6, 14), (7, 15), (8, 16)])
    cir += shift(ModMulti1xmod15Rev(), ns)

    cir += X(4, 3)
    return cir
