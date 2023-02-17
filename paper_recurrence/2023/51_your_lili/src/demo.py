"""Demo code for arithmetic circuits."""

from src.circuit import Adder, AdderRev, ModAdder, ModAdderRev
from src.circuit import ModMulti7xmod15, ModExp
from src.utils import c2q, q2c


def demo1():
    """验证 Fig.6（加法器）, Fig.7 （减法器）
    """
    print("验证加法器,减法器:")
    # 验证 b + a, b - a
    a = 3
    b = 8
    cir_a = c2q([3, 2, 1, 0], a)
    cir_b = c2q([7, 6, 5, 4], b)

    cir_add = cir_a + cir_b + Adder()
    q_add = q2c(cir_add, [7, 6, 5, 4])

    cir_sub = cir_a + cir_b + AdderRev()
    q_sub = q2c(cir_sub, [7, 6, 5, 4])

    print(
        f"q_add = {q_add}, c_add = {b + a}\nq_sub = {q_sub}, c_sub = {b - a}\n")


def demo2():
    """验证模加法器、模减法器
    """
    print("验证模加法器、模减法器:")
    # 1. 计算 (a + b) mod n
    # 2. 计算 (b - a) mod n。注：对 b<a, (b-a) mod n = (b-a+n) mod n
    a = 11
    b = 9
    n = 15
    # 将经典数值转换成量子线路编码
    cir_a = c2q([3, 2, 1, 0], a)
    cir_b = c2q([7, 6, 5, 4], b)
    cir_n = c2q([16, 15, 14, 13], n)  # 输入置1

    cir_madd = cir_a + cir_b + cir_n + ModAdder() + cir_n
    cir_msub = cir_a + cir_b + cir_n + ModAdderRev() + cir_n
    # 使用量子线路计算结果
    q_add = q2c(cir_madd, [7, 6, 5, 4])
    q_sub = q2c(cir_msub, [7, 6, 5, 4])
    # 经典计算验证
    c_add = (a + b) % n
    c_sub = (b - a) % n

    print(
        f"q_add = {q_add}, c_add = {c_add}\nq_sub = {q_sub}, c_sub = {c_sub}\n")


def demo3():
    """验证模乘法器
    """
    print("验证模乘法器(8U32G硬件大约耗时20秒):")
    # 计算 7x mod n
    ctrl = 1  # 控制位
    x = 9
    n = 15
    cir_ctrl = c2q([0], ctrl)
    cir_x = c2q([4, 3, 2, 1], x)
    cir_n = c2q([21, 20, 19, 18], n)
    cir_7x = cir_ctrl + cir_x + cir_n + ModMulti7xmod15() + cir_n
    # 获取终态得到结果
    q_mul = q2c(cir_7x, [12, 11, 10, 9])
    c_mul = (7 * x) % 15

    print(f"q_mul = {q_mul}, c_mul = {c_mul}\n")


def demo4():
    """验证模指数器
    """
    # 计算 7^a mod n
    print("验证模指数器(8U32G硬件大约耗时20分钟):")
    a = 3
    n = 15
    cir_a = c2q([3, 2, 1, 0], a)
    cir_n = c2q([25, 24, 23, 22], n)
    cir_expmod = cir_a + cir_n + ModExp() + cir_n

    q_res = q2c(cir_expmod, [8, 7, 6, 5])
    c_res = (7**a) % n

    print(f"q_res = {q_res}, c_res = {c_res}\n")


if __name__ == "__main__":
    demo1()
    demo2()
    demo3()
    demo4()
