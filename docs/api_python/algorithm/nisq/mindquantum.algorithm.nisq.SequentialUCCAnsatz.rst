mindquantum.algorithm.nisq.SequentialUCCAnsatz
================================================

.. py:class:: mindquantum.algorithm.nisq.SequentialUCCAnsatz(n_qubits=None, n_electrons=None)

    顺序幺正耦合簇（Sequential UCC）ansatz。

    ``SequentialUCCAnsatz`` 通过顺序追加 :class:`~.core.operators.FermionOperator` 来构建UCC线路，不会像常规费米子算符那样合并重复激发项，因此适合多层或Trotter化场景中“同一激发项重复出现但参数不同”的构造方式。

    .. note::
        该ansatz内部使用 :class:`~.simulator.mqchem.UCCExcitationGate` 构建线路。
        生成的线路应与 :class:`~.simulator.mqchem.MQChemSimulator` 配合使用，不适用于标准态矢量 :class:`~.simulator.Simulator`。

    参数：
        - **n_qubits** (int) - 模拟中的量子比特（自旋轨道）数量，要求大于等于 ``1``。默认值： ``None``。
        - **n_electrons** (int) - 电子数，要求大于等于 ``0`` 且不大于 ``n_qubits``。默认值： ``None``。

    .. py:method:: append(operator)

        在ansatz末尾追加一个激发算符，并立即更新内部线路。

        参数：
            - **operator** (FermionOperator) - 待追加的费米子激发算符。该算符会先进行规范化与合法性检查：
              必须是单项、非恒等项、且单项内不允许重复轨道索引；若设置了 ``n_qubits``，算符作用索引不得越界。

        异常：
            - **TypeError** - `operator` 不是 :class:`~.core.operators.FermionOperator`。
            - **ValueError** - `operator` 不满足项数、索引唯一性或边界约束。

    .. py:method:: remove(index)

        删除并返回给定索引处的算符，同时重建内部线路。

        参数：
            - **index** (int) - 需要删除的算符索引。

        返回：
            FermionOperator，删除算符的副本。

        异常：
            - **TypeError** - `index` 不是整数。
            - **IndexError** - `index` 越界。

    .. py:method:: operators
        :property:

        返回当前算符列表的副本。

        返回：
            list[FermionOperator]，内部算符的拷贝列表。对返回列表或其中元素的修改，不会影响ansatz内部状态。
