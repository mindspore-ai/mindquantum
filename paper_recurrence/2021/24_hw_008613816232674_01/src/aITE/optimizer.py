from .Hessian.gradients import Grad, FisherInformation
from .Hessian.utils import pr2array, array2pr
import numpy as np

class SGD:
    def __init__(self, learn_rate=10, decay_rate=0.01):
        self.diff = np.zeros(1).astype(np.float32)
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        
    def step(self, vector, grad):
        # Performing the gradient descent loop
        vector = vector.astype(np.float32)
        # grad = grad.squeeze(0).squeeze(0)
        grad = grad.astype(np.float32)

        self.diff = self.decay_rate * self.diff - self.learn_rate * grad
        vector += self.diff
        return vector


class Optimizer:
    def __init__(self, circ, pr, n_qubits, opt_type='aite', lr=0.01):

        self.circ, self.pr, self.n_qubits = circ, pr, n_qubits
        self.opt_type = opt_type

        self.optimizer = SGD(learn_rate=lr)

    def preconditioner(self, pr):
        preconditioner = FisherInformation(self.circ, pr, self.n_qubits)
        if self.opt_type=='aite':
            hess = preconditioner.gite_preconditional()
        elif self.opt_type=='qngd':
            hess = preconditioner.fisher_information()
        elif self.opt_type=='gd':
            hess = None
        else:
            raise ValueError()
        return hess

    def step(self, ham, root_gradient=None):
        parameters, k_list = pr2array(self.pr)

        g = Grad(self.circ, self.pr, ham, self.n_qubits).grad_reserveMode()

        if root_gradient is not None:
            g = g * root_gradient

        if self.opt_type!='gd':
            h = self.preconditioner(self.pr)
            g = np.linalg.inv(h + np.eye(len(h))*1e-15).dot(g[:, np.newaxis]).squeeze(1) * (-1)

        parameters = self.optimizer.step(parameters, g).real
        self.pr = array2pr(parameters, k_list)