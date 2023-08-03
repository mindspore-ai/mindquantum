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

以下Ansatz来源于论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_.

.. toctree::
    :hidden:

    mindquantum.algorithm.nisq.Ansatz1
    mindquantum.algorithm.nisq.Ansatz2
    mindquantum.algorithm.nisq.Ansatz3
    mindquantum.algorithm.nisq.Ansatz4
    mindquantum.algorithm.nisq.Ansatz5
    mindquantum.algorithm.nisq.Ansatz6
    mindquantum.algorithm.nisq.Ansatz7
    mindquantum.algorithm.nisq.Ansatz8
    mindquantum.algorithm.nisq.Ansatz9
    mindquantum.algorithm.nisq.Ansatz10
    mindquantum.algorithm.nisq.Ansatz11
    mindquantum.algorithm.nisq.Ansatz12
    mindquantum.algorithm.nisq.Ansatz13
    mindquantum.algorithm.nisq.Ansatz14
    mindquantum.algorithm.nisq.Ansatz15
    mindquantum.algorithm.nisq.Ansatz16
    mindquantum.algorithm.nisq.Ansatz17
    mindquantum.algorithm.nisq.Ansatz18
    mindquantum.algorithm.nisq.Ansatz19

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Class
      - Images
    * - :class:`mindquantum.algorithm.nisq.Ansatz1`
      - .. image:: ./ansatz_images/ansatz1.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz2`
      - .. image:: ./ansatz_images/ansatz2.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz3`
      - .. image:: ./ansatz_images/ansatz3.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz4`
      - .. image:: ./ansatz_images/ansatz4.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz5`
      - .. image:: ./ansatz_images/ansatz5.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz6`
      - .. image:: ./ansatz_images/ansatz6.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz7`
      - .. image:: ./ansatz_images/ansatz7.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz8`
      - .. image:: ./ansatz_images/ansatz8.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz9`
      - .. image:: ./ansatz_images/ansatz9.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz10`
      - .. image:: ./ansatz_images/ansatz10.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz11`
      - .. image:: ./ansatz_images/ansatz11.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz12`
      - .. image:: ./ansatz_images/ansatz12.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz13`
      - .. image:: ./ansatz_images/ansatz13.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz14`
      - .. image:: ./ansatz_images/ansatz14.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz15`
      - .. image:: ./ansatz_images/ansatz15.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz16`
      - .. image:: ./ansatz_images/ansatz16.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz17`
      - .. image:: ./ansatz_images/ansatz17.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz18`
      - .. image:: ./ansatz_images/ansatz18.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz19`
      - .. image:: ./ansatz_images/ansatz19.png
            :height: 180px

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
    mindquantum.algorithm.nisq.ansatz_variance
