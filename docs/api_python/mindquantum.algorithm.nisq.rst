mindquantum.algorithm.nisq
===========================

.. py:module:: mindquantum.algorithm.nisq


NISQ算法。

Base Class
-------------

.. mscnautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.Ansatz

Encoder
-------------

.. mscnautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.IQPEncoding

Ansatz
-------------

.. mscnautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.HardwareEfficientAnsatz
    mindquantum.algorithm.nisq.Max2SATAnsatz
    mindquantum.algorithm.nisq.MaxCutAnsatz
    mindquantum.algorithm.nisq.QubitUCCAnsatz
    mindquantum.algorithm.nisq.StronglyEntangling
    mindquantum.algorithm.nisq.UCCAnsatz

Generator
-------------

.. mscnautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.generate_uccsd
    mindquantum.algorithm.nisq.quccsd_generator
    mindquantum.algorithm.nisq.uccsd0_singlet_generator
    mindquantum.algorithm.nisq.uccsd_singlet_generator

Functional
-------------

.. mscnautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.Transform
    mindquantum.algorithm.nisq.get_qubit_hamiltonian
    mindquantum.algorithm.nisq.uccsd_singlet_get_packed_amplitudes
