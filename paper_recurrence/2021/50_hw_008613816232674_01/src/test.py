import numpy as np
from Hessian.helper import Gradient_test
from Hessian.utils import Ising_like_ham
from Hessian.gradients import Grad 
from timeit import default_timer

n_qubits = 3
Ham = Ising_like_ham(n_qubits)
ham = Ham.local_Hamiltonian()

gt = Gradient_test(n_qubits)
gsfm = Grad(gt.circ, gt.pr, ham, gt.n_qubits)


def test_grad():
    jac, hess = gsfm.grad()
    jac_helper, hess_helper = gt.gradient(ham)
    g = gsfm.grad_reserveMode()
    
    print(jac)
    print(g)

    print('max diff: jac_forward vs jac_helper,', np.abs(jac - jac_helper).max())
    print('max diff: jac_forward vs jac_reverse,', np.abs(jac - g).max())
    print('max diff: jac_reverse vs jac_helper,', np.abs(g - jac_helper).max())

def test_hess():
    end = default_timer()
    jac, hess = gsfm.grad()
    hess_time_usage = default_timer() - end 

    end = default_timer()
    jac_helper, hess_helper = gt.gradient(ham)
    hess_helper_time_usage = default_timer() - end


    end = default_timer()
    hess_forward = gsfm.Hess_forwardMode()
    hess_forward_time_usage = default_timer()-end

    print(hess)
    print(hess_forward)
#     raise ValueError()

    print('max diff: hess_hybrid vs hess_hybrid_helper,', np.abs(hess - hess_helper).max())
    print('max diff: hess_hybrid vs hess_forward,', np.abs(hess - hess_forward).max())
    print('max diff: hess_hybrid_helper vs hess_forward,', np.abs(hess_helper - hess_forward).max())

    print('time usage: hess_hybrid {}s'.format(hess_time_usage))
    print('time usage: hess_hybrid_helper {}s'.format(hess_helper_time_usage))
    print('time usage: hess_forward {}s'.format(hess_forward_time_usage))

if __name__ == '__main__':
    test_grad()
    test_hess()

    # F = FisherInformation(gt.circ, gt.pr, gt.n_qubits)
    # fisher = F.fisherInformation_hybridMode()
    # fisher_builtin = F.fisherInformation_builtin()
    # print(fisher)
    # print(fisher_builtin)