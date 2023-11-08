import mindspore as ms
import numpy as np
import os
import argparse
from mindspore import nn
from mindspore.amp import all_finite
from src.dense_ode_net import DenseODENet, DenseODENetWithLoss
from src.data_generate import Generator
from utils import init_model, generate_train_data, init_env, makedir, get_config, evaluate, test


ms.set_seed(1999)
np.random.seed(1999)


def single_train(config, time_step: int, net_with_loss: DenseODENetWithLoss, data_generator):
    lr = config['lr'] * np.power(config['lr_reduce_gamma'], (time_step - 1) // config['lr_reduce_interval'])
    my_optimizer = nn.Adam(params=net_with_loss.ode_net.trainable_params(), learning_rate=lr)

    def forward_fn(h, trajectory, t_points):
        loss = net_with_loss.get_loss(H=h, batch_trajectories=trajectory, t_points=t_points)
        return loss
    value_and_grad = ms.ops.value_and_grad(forward_fn, None, weights=my_optimizer.parameters)

    def train_process(h, trajectory, t_points):
        loss, grads = value_and_grad(h, trajectory, t_points)
        if all_finite(grads):
            my_optimizer(grads)
        return loss

    train_dataset = generate_train_data(config=config, data_generator=data_generator, time_step=time_step)
    for epoch_idx in range(1, config['epochs'] + 1):
        net_with_loss.ode_net.set_train(mode=True)
        avg_loss = 0
        for H, trajectories, time_points in train_dataset.fetch():
            train_loss = train_process(H, trajectories, time_points)
            avg_loss += train_loss.asnumpy()
        print('time_step: {} -- epoch: {} -- lr: {} -- loss: {}'.format(time_step, epoch_idx, lr, avg_loss))

        # generate new data
        if epoch_idx % config['generate_data_interval'] == 0:
            train_dataset = generate_train_data(config=config, data_generator=data_generator, time_step=time_step)
    # save
    save_path = os.path.join(config['save_directory'], 'dense_ode_net_step{}.ckpt'.format(time_step))
    ms.save_checkpoint(net_with_loss.ode_net, save_path)
    # evaluate
    evaluate(config=config, model=net_with_loss.ode_net, data_generator=data_generator)
    print('-- Current Dense Weight')
    print(net_with_loss.ode_net.dense_w())
    return


def train(config):
    data_generator = Generator(dim=config['h_dim'])
    dense_ode_net = init_model(config)
    net_with_loss = DenseODENetWithLoss(ode_net=dense_ode_net)
    # first evaluate
    evaluate(config=config, model=net_with_loss.ode_net, data_generator=data_generator)
    print('-- Current Dense Weight')
    print(net_with_loss.ode_net.dense_w())

    for time_step in range(1, config['time_step_num'] + 1):
        single_train(config=config, time_step=time_step, net_with_loss=net_with_loss, data_generator=data_generator)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dense ODE-Net for schrodinger train")
    parser.add_argument("--mode", type=str, default="PYNATIVE", choices=["PYNATIVE"], help="Running in PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="CPU", choices=["CPU"],
                        help="The target device to run, support 'CPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()
    init_env(env_args=args)

    my_config = get_config(args.config_file_path)
    my_config['device_target'] = args.device_target
    my_config['context_mode'] = args.mode
    my_config['device_id'] = args.device_id

    makedir(my_config)
    train(my_config)
    test(my_config)
