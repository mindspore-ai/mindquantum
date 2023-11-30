mindquantum.algorithm.nisq
===========================

.. automodule:: mindquantum.algorithm.nisq

Base Class
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.Ansatz

Encoder
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.IQPEncoding

Ansatz
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.HardwareEfficientAnsatz
    mindquantum.algorithm.nisq.Max2SATAnsatz
    mindquantum.algorithm.nisq.MaxCutAnsatz
    mindquantum.algorithm.nisq.QubitUCCAnsatz
    mindquantum.algorithm.nisq.StronglyEntangling
    mindquantum.algorithm.nisq.UCCAnsatz

.. toctree::
    :hidden:

    nisq/mindquantum.algorithm.nisq.RYLinear
    nisq/mindquantum.algorithm.nisq.RYFull
    nisq/mindquantum.algorithm.nisq.RYCascade
    nisq/mindquantum.algorithm.nisq.RYRZFull
    nisq/mindquantum.algorithm.nisq.PCHeaXYZ1F
    nisq/mindquantum.algorithm.nisq.PCHeaXYZ2F
    nisq/mindquantum.algorithm.nisq.ASWAP

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Class
      - Images
    * - :class:`mindquantum.algorithm.nisq.RYLinear`
      - .. image:: nisq/ansatz_images/RYLinear.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.RYFull`
      - .. image:: nisq/ansatz_images/RYFull.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.RYCascade`
      - .. image:: nisq/ansatz_images/RYCascade.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.RYRZFull`
      - .. image:: nisq/ansatz_images/RYRZFull.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.PCHeaXYZ1F`
      - .. image:: nisq/ansatz_images/PCHeaXYZ1F.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.PCHeaXYZ2F`
      - .. image:: nisq/ansatz_images/PCHeaXYZ2F.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.ASWAP`
      - .. image:: nisq/ansatz_images/ASWAP.png
            :height: 180px

The following Ansatz come from paper `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_.

.. toctree::
    :hidden:

    nisq/mindquantum.algorithm.nisq.Ansatz1
    nisq/mindquantum.algorithm.nisq.Ansatz2
    nisq/mindquantum.algorithm.nisq.Ansatz3
    nisq/mindquantum.algorithm.nisq.Ansatz4
    nisq/mindquantum.algorithm.nisq.Ansatz5
    nisq/mindquantum.algorithm.nisq.Ansatz6
    nisq/mindquantum.algorithm.nisq.Ansatz7
    nisq/mindquantum.algorithm.nisq.Ansatz8
    nisq/mindquantum.algorithm.nisq.Ansatz9
    nisq/mindquantum.algorithm.nisq.Ansatz10
    nisq/mindquantum.algorithm.nisq.Ansatz11
    nisq/mindquantum.algorithm.nisq.Ansatz12
    nisq/mindquantum.algorithm.nisq.Ansatz13
    nisq/mindquantum.algorithm.nisq.Ansatz14
    nisq/mindquantum.algorithm.nisq.Ansatz15
    nisq/mindquantum.algorithm.nisq.Ansatz16
    nisq/mindquantum.algorithm.nisq.Ansatz17
    nisq/mindquantum.algorithm.nisq.Ansatz18
    nisq/mindquantum.algorithm.nisq.Ansatz19

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Class
      - Images
    * - :class:`mindquantum.algorithm.nisq.Ansatz1`
      - .. image:: nisq/ansatz_images/ansatz1.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz2`
      - .. image:: nisq/ansatz_images/ansatz2.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz3`
      - .. image:: nisq/ansatz_images/ansatz3.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz4`
      - .. image:: nisq/ansatz_images/ansatz4.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz5`
      - .. image:: nisq/ansatz_images/ansatz5.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz6`
      - .. image:: nisq/ansatz_images/ansatz6.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz7`
      - .. image:: nisq/ansatz_images/ansatz7.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz8`
      - .. image:: nisq/ansatz_images/ansatz8.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz9`
      - .. image:: nisq/ansatz_images/ansatz9.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz10`
      - .. image:: nisq/ansatz_images/ansatz10.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz11`
      - .. image:: nisq/ansatz_images/ansatz11.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz12`
      - .. image:: nisq/ansatz_images/ansatz12.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz13`
      - .. image:: nisq/ansatz_images/ansatz13.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz14`
      - .. image:: nisq/ansatz_images/ansatz14.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz15`
      - .. image:: nisq/ansatz_images/ansatz15.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz16`
      - .. image:: nisq/ansatz_images/ansatz16.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz17`
      - .. image:: nisq/ansatz_images/ansatz17.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz18`
      - .. image:: nisq/ansatz_images/ansatz18.png
            :height: 180px
    * - :class:`mindquantum.algorithm.nisq.Ansatz19`
      - .. image:: nisq/ansatz_images/ansatz19.png
            :height: 180px

Generator
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.generate_uccsd
    mindquantum.algorithm.nisq.quccsd_generator
    mindquantum.algorithm.nisq.uccsd0_singlet_generator
    mindquantum.algorithm.nisq.uccsd_singlet_generator

Functional
-----------

.. autosummary::
    :toctree: nisq
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.algorithm.nisq.Transform
    mindquantum.algorithm.nisq.get_qubit_hamiltonian
    mindquantum.algorithm.nisq.uccsd_singlet_get_packed_amplitudes
    mindquantum.algorithm.nisq.ansatz_variance
    mindquantum.algorithm.nisq.get_reference_circuit
