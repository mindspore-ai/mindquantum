from mindquantum.framework import MQAnsatzOnlyLayer
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
import copy

def Generate_iterations(one_step_iteration_generator,several_steps_iteration_generator,x_init,num_iters,*args):
    if callable(one_step_iteration_generator):
        iterations_generated,value_iterations_generated = [],[]
        x_now = x_init
        for _ in range(num_iters):
            x_next,value_now = one_step_iteration_generator(x_now,*args)
            iterations_generated.append(x_next)
            value_iterations_generated.append(value_now)
            x_now = x_next
    else:
        iterations_generated,value_iterations_generated = several_steps_iteration_generator(x_init,num_iters,*args)

    return iterations_generated,value_iterations_generated


def ising_vqe_its_generator(x_init,num_iters,optimizer_fn,lr,simulator,ham,ansatz):
  x_init = x_init.real.astype(np.float32)
  grad_ops = simulator.get_expectation_with_grad(ham,ansatz)
  net = MQAnsatzOnlyLayer(grad_ops,Tensor(x_init))  
  opti = optimizer_fn(net.trainable_params(),learning_rate=lr)
  train_net = nn.TrainOneStepCell(net,opti)

  params = []
  values = []

  for i in range(num_iters):
    values.append(train_net().asnumpy())
    param = np.array(net.weight.asnumpy())
    # param = copy.deepcopy(net.weight.asnumpy())
    # param = net.weight.asnumpy()
    params.append(param)

  return params,values