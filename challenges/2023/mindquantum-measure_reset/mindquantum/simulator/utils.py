# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simulator utils."""


def _thread_balance(n_prs, n_meas, parallel_worker):
    """Thread balance."""
    if parallel_worker is None:
        parallel_worker = n_meas * n_prs
    if n_meas * n_prs <= parallel_worker:
        batch_threads = n_prs
        mea_threads = n_meas
    else:
        if n_meas < n_prs:
            batch_threads = min(n_prs, parallel_worker)
            mea_threads = min(n_meas, max(1, parallel_worker // batch_threads))
        else:
            mea_threads = min(n_meas, parallel_worker)
            batch_threads = min(n_prs, max(1, parallel_worker // mea_threads))
    return batch_threads, mea_threads


class GradOpsWrapper:  # pylint: disable=too-many-instance-attributes
    """
    Wrapper the gradient operator that with the information that generate this gradient operator.

    Args:
        grad_ops (Union[FunctionType, MethodType]): A function or a method
            that return forward value and gradient w.r.t parameters.
        hams (Hamiltonian): The hamiltonian that generate this grad ops.
        circ_right (Circuit): The right circuit that generate this grad ops.
        circ_left (Circuit): The left circuit that generate this grad ops.
        encoder_params_name (list[str]): The encoder parameters name.
        ansatz_params_name (list[str]): The ansatz parameters name.
        parallel_worker (int): The number of parallel worker to run the batch.
        sim (Simulator): The simulator that this grad ops used.
    """

    def __init__(
        self, grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name, parallel_worker, sim=None
    ):  # pylint: disable=too-many-arguments
        """Initialize a GradOpsWrapper object."""
        self.grad_ops = grad_ops
        self.hams = hams
        self.circ_right = circ_right
        self.circ_left = circ_left
        self.encoder_params_name = encoder_params_name
        self.ansatz_params_name = ansatz_params_name
        self.parallel_worker = parallel_worker
        self.str = ''
        self.sim = sim

    def __call__(self, *args):
        """Definition of a function call operator."""
        return self.grad_ops(*args)

    def set_str(self, grad_str):
        """
        Set expression for gradient operator.

        Args:
            grad_str (str): The string of QNN operator.
        """
        self.str = grad_str
