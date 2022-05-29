.. py:method:: mindquantum.simulator.Simulator.get_expectation_with_grad(hams, circ_right, circ_left=None, simulator_left=None, encoder_params_name=None, ansatz_params_name=None, parallel_worker=None)

    获取一个返回前向值和关于线路参数梯度的函数。该方法旨在计算期望值及其梯度，如下所示。

    .. math::

        E = \left<\varphi\right|U_l^\dagger H U_r \left|\psi\right>

    其中 :math:`U_l` 是circ_left，:math:`U_r` 是circ_right，:math:`H` 是hams, :math:`\left|\psi\right>` 是模拟器当前的量子态, :math:`\left|\varphi\right>` 是 `simulator_left` 的量子态。

    **参数：**

    - **hams** (Hamiltonian) – 需要计算期望的Hamiltonian。
    - **circ_right** (Circuit) – 上述 :math:`U_r` 电路。
    - **circ_left** (Circuit) – 上述 :math:`U_l` 电路，默认情况下，这个线路将为None，在这种情况下， :math:`U_l` 将等于 :math:`U_r` 。默认值：None。
    - **simulator_left** (Simulator) – 包含 :math:`\left|\varphi\right>` 的模拟器。如果无，则 :math:`\left|\varphi\right>` 被假定等于 :math:`\left|\psi\right>`。默认值：None。
    - **encoder_params_name** (list[str]) – 指定哪些参数属于encoder，被编码成量子态。encoder数据可以是一个batch。默认值：None。
    - **ansatz_params_name** (list[str]) – 指定哪些参数属于ansatz，被在训练期间训练。默认值：None。
    - **parallel_worker** (int) – 并行器数目。并行器可以在并行线程中处理batch。默认值：None。

    **返回：**

    GradOpsWrapper，一个包含生成梯度算子信息的梯度算子包装器。