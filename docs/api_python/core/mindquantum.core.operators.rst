mindquantum.core.operators
==========================

.. py:module:: mindquantum.core.operators


MindQuantum算子库。算子由一个或多个基本门的组合而成。

包含以下类的表示：

- Qubit算子

- Fermion算子

- 时间演化算子

Class
---------------

.. mscnautosummary::
    :toctree: operators
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.operators.FermionOperator
    mindquantum.core.operators.Hamiltonian
    mindquantum.core.operators.InteractionOperator
    mindquantum.core.operators.PolynomialTensor
    mindquantum.core.operators.Projector
    mindquantum.core.operators.QubitExcitationOperator
    mindquantum.core.operators.QubitOperator
    mindquantum.core.operators.TimeEvolution

Function
---------------

.. mscnautosummary::
    :toctree: operators
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.operators.commutator
    mindquantum.core.operators.count_qubits
    mindquantum.core.operators.down_index
    mindquantum.core.operators.get_fermion_operator
    mindquantum.core.operators.ground_state_of_sum_zz
    mindquantum.core.operators.hermitian_conjugated
    mindquantum.core.operators.normal_ordered
    mindquantum.core.operators.number_operator
    mindquantum.core.operators.sz_operator
    mindquantum.core.operators.up_index
