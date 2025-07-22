mindquantum.simulator
=====================

.. py:module:: mindquantum.simulator


模拟量子系统演化的量子模拟器。

Class
-------

.. mscnautosummary::
    :toctree: simulator
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.simulator.GradOpsWrapper
    mindquantum.simulator.Simulator
    mindquantum.simulator.NoiseBackend
    mindquantum.simulator.mqchem.MQChemSimulator
    mindquantum.simulator.mqchem.CIHamiltonian
    mindquantum.simulator.mqchem.UCCExcitationGate

Function
---------

.. mscnautosummary::
    :toctree: simulator
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.simulator.fidelity
    mindquantum.simulator.get_supported_simulator
    mindquantum.simulator.inner_product
    mindquantum.simulator.get_stabilizer_string
    mindquantum.simulator.get_tableau_string
    mindquantum.simulator.decompose_stabilizer
    mindquantum.simulator.mqchem.prepare_uccsd_vqe
