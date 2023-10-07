mindquantum.algorithm.nisq.QubitUCCAnsatz
==========================================

.. py:class:: mindquantum.algorithm.nisq.QubitUCCAnsatz(n_qubits=None, n_electrons=None, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1)

    量子比特幺正耦合簇（qUCC）是幺正耦合簇的变体，它使用量子比特激励算子而不是费米子激励算子。
    量子比特激励算子跨越的Fock空间相当于费米子算子，因此可以使用量子比特激发算子以更高阶的 Trotterization 为代价来近似精确的波函数。

    qUCC最大的优点是：即使使用3阶或4阶Trotterization，CNOT门的数量比UCC的原始版本小得多。
    此外，尽管变分参数的数量增加，但精度也大大提高。

    .. note::
        不包括哈特里-福克电路。
        目前，不允许generalized=True，因为需要理论验证。
        参考文献： `Efficient quantum circuits for quantum computational chemistry <https://doi.org/10.1103/PhysRevA.102.062612>`_。

    参数：
        - **n_qubits** (int) - 模拟中量子比特（自旋轨道）的数量。默认值： ``None``。
        - **n_electrons** (int) - 给定分子的电子数。默认值： ``None``。
        - **occ_orb** (list) - 手动分配的占用空间轨道的索引。默认值： ``None``。
        - **vir_orb** (list) - 手动分配的虚拟空间轨道的索引。默认值： ``None``。
        - **generalized** (bool) - 是否使用不区分占用轨道或虚拟轨道的广义激励（qUCCGSD）。目前，不允许 `generalized=True` ，因为需要理论验证。默认值： ``False``。
        - **trotter_step** (int) - 梯度的数量。建议设置大于等于2的值，以获得较好的精度。默认值： ``1``。
