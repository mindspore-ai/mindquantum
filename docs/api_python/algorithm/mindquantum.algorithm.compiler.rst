mindquantum.algorithm.compiler
==============================

.. py:module:: mindquantum.algorithm.compiler


MindQuantum 量子线路编译模块。

Fixed decompose rules
---------------------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.compiler.ch_decompose
    mindquantum.algorithm.compiler.crx_decompose
    mindquantum.algorithm.compiler.crxx_decompose
    mindquantum.algorithm.compiler.cry_decompose
    mindquantum.algorithm.compiler.cryy_decompose
    mindquantum.algorithm.compiler.cswap_decompose
    mindquantum.algorithm.compiler.ct_decompose
    mindquantum.algorithm.compiler.cy_decompose
    mindquantum.algorithm.compiler.cz_decompose
    mindquantum.algorithm.compiler.rxx_decompose
    mindquantum.algorithm.compiler.ryy_decompose
    mindquantum.algorithm.compiler.rzz_decompose
    mindquantum.algorithm.compiler.cs_decompose
    mindquantum.algorithm.compiler.swap_decompose
    mindquantum.algorithm.compiler.ccx_decompose

Universal decompose rules
-------------------------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.compiler.euler_decompose
    mindquantum.algorithm.compiler.cu_decompose
    mindquantum.algorithm.compiler.qs_decompose
    mindquantum.algorithm.compiler.abc_decompose
    mindquantum.algorithm.compiler.kak_decompose
    mindquantum.algorithm.compiler.tensor_product_decompose

Compiler rules
--------------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.compiler.BasicCompilerRule
    mindquantum.algorithm.compiler.KroneckerSeqCompiler
    mindquantum.algorithm.compiler.SequentialCompiler
    mindquantum.algorithm.compiler.BasicDecompose
    mindquantum.algorithm.compiler.CZBasedChipCompiler
    mindquantum.algorithm.compiler.CXToCZ
    mindquantum.algorithm.compiler.CZToCX
    mindquantum.algorithm.compiler.GateReplacer
    mindquantum.algorithm.compiler.FullyNeighborCanceler
    mindquantum.algorithm.compiler.SimpleNeighborCanceler
    mindquantum.algorithm.compiler.compile_circuit

DAG circuit
-----------

.. mscnautosummary::
    :toctree: compiler
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.compiler.DAGCircuit
    mindquantum.algorithm.compiler.DAGNode
    mindquantum.algorithm.compiler.GateNode
    mindquantum.algorithm.compiler.DAGQubitNode
    mindquantum.algorithm.compiler.connect_two_node
    mindquantum.algorithm.compiler.try_merge
