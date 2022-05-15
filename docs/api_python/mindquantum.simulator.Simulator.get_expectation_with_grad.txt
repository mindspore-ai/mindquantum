mindquantum.simulator.Simulator.get_expectation_with_grad(hams, circ_right, circ_left=None, simulator_left=None, encoder_params_name=None, ansatz_params_name=None, parallel_worker=None)

        获取一个返回正向值和梯度w.r.t电路参数的函数。
        此方法旨在计算期望及其梯度，如下所示。

        .. math::

            E = \left<\varphi\right|U_l^\dagger H U_r \left|\psi\right>

        其中 :math:`U_l` 是迂回左，:math:`U_r` 是迂回右，:math:`H` 是火腿
        和 :math:`\left|\psi\right>` 是此模拟器的当前量子状态，
        和 :math:`\left|\varphi\right>` 是`模拟器_左'的量子状态。

        参数:
            hams (Hamiltonian): 需要得到期望的汉密尔顿式。
            circ_right (Circuit): 上述 :math:`U_r` 电路。
            circ_left (Circuit): 上述 :math:`U_l` 电路，默认情况下，此电路将为无，在这种情况下， :math:`U_l` 将等于 :math:`U_r`. 默认值：None。
            simulator_left (Simulator): 包含 :math:`\left|\varphi\right>` 的模拟器。如果无，则 :math:`\left|\varphi\right>` 被假定等于 :math:`\left|\psi\right>`。默认值：None。
            encoder_params_name (list[str]): 要指定哪些参数属于编码器，将输入数据编码为量子状态。编码器数据可以是批处理。默认值：None。
            ansatz_params_name (list[str]): 具体哪些参数属于nsatz，将在训练期间训练。默认值：None。
            parallel_worker (int): 并行工作器编号。并行工作程序可以在并行线程中处理批处理。默认值：None。

        返回:
            格拉德操作包装器，一个格拉德操作包装器，而不是包含生成此格拉德操作的信息。

        样例:
            >>> import numpy as np
            >>> from mindquantum import Simulator, Hamiltonian
            >>> from mindquantum import Circuit
            >>> from mindquantum.core.operators import QubitOperator
            >>> circ = Circuit().ry('a', 0)
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> sim = Simulator('projectq', 1)
            >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
            >>> grad_ops(np.array([1.0]))
            (array([[0.54030231+0.j]]), array([[[-0.84147098+0.j]]]))
            >>> sim1 = Simulator('projectq', 1)
            >>> prep_circ = Circuit().h(0)
            >>> ansatz = Circuit().ry('a', 0).rz('b', 0).ry('c', 0)
            >>> sim1.apply_circuit(prep_circ)
            >>> sim2 = Simulator('projectq', 1)
            >>> ham = Hamiltonian(QubitOperator(""))
            >>> grad_ops = sim2.get_expectation_with_grad(ham, ansatz, Circuit(), simulator_left=sim1)
            >>> f, g = grad_ops(np.array([7.902762e-01, 2.139225e-04, 7.795934e-01]))
            >>> f
            array([[0.99999989-7.52279618e-05j]])
        