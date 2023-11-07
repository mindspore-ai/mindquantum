import mindspore as ms
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.data_generate import Generator, Dataset
from src.dense_ode_net import DenseODENet


def get_config(config_path):
    with open(config_path, 'r') as c_f:
        config = yaml.safe_load(c_f)
    return config


def init_env(env_args):
    ms.set_context(mode=ms.GRAPH_MODE if env_args.mode.upper().startswith("GRAPH") else ms.PYNATIVE_MODE,
                   save_graphs=env_args.save_graphs,
                   save_graphs_path=env_args.save_graphs_path,
                   device_target=env_args.device_target,
                   device_id=env_args.device_id)
    print('----------------------------- Env settings -----------------------------')
    print('running on {}, device id: {}, context mode: {}'.format(env_args.device_target, env_args.device_id,
                                                                  env_args.mode))
    return


# def init_env(config):
#     ms.set_context(mode=ms.GRAPH_MODE if config['mode'].upper().startswith("GRAPH") else ms.PYNATIVE_MODE,
#                    save_graphs=False, save_graphs_path='./graphs', device_target=config['device_target'],
#                    device_id=config['device_id'])
#     print('----------------------------- Env settings -----------------------------')
#     print('running on {}, device id: {}, context mode: {}'.format(config['device_target'], config['device_id'],
#                                                                   config['mode']))
#     return


def makedir(config):
    if not os.path.exists(config['save_directory']):
        os.mkdir(config['save_directory'])
    if not os.path.exists(config['data_directory']):
        os.mkdir(config['data_directory'])
    if not os.path.exists(config['images_directory']):
        os.mkdir(config['images_directory'])
    return


def init_model(config):
    model = DenseODENet(depth=config['depth'], h_dim=config['h_dim'], init_range=config['w_init_range'],
                        max_dt=config['max_dt'])
    return model


def generate_train_data(config, data_generator: Generator, time_step: int):
    dt = config['generate_dt']
    t_list = np.arange(dt, dt * (time_step + 1), dt)
    h_list, trajectories, time_points = data_generator.generate_multi_h_trajectory(h_num=config['generate_h_num'],
                                                                                   s0_num=config['generate_init_num'],
                                                                                   t_list=t_list)
    train_dataset = Dataset(H_list=h_list, trajectories=trajectories, t_points=time_points,
                            batch_size=config['train_batch_size'], shuffle=True, dtype=ms.float32)
    return train_dataset


def random_t_list(max_t, t_num, max_dt):
    t_list = np.random.rand(t_num) * max_t
    t_list = np.sort(t_list)
    for idx in range(t_list.shape[0]):
        t_list[idx] += (idx + 1) * max_dt
    return t_list


def generate_test_data(data_generator: Generator, config, h_num, init_num, flag: str):
    if flag.upper() == 'EVALUATE':
        max_t = config['evaluate_max_t']
        t_num = config['evaluate_t_points_num']
        t_list = random_t_list(max_t=max_t, t_num=t_num, max_dt=config['max_dt'])
    else:
        max_t = config['test_max_t']
        t_num = config['test_t_points_num']
        dt = max_t / t_num
        t_list = np.arange(dt, dt * (t_num + 1), dt)
    h_list, trajectories, time_points = data_generator.generate_multi_h_trajectory(h_num=h_num, s0_num=init_num,
                                                                                   t_list=t_list)
    test_dataset = Dataset(H_list=h_list, trajectories=trajectories, t_points=time_points,
                           batch_size=init_num, shuffle=False, dtype=ms.float32)
    return test_dataset


def relative_l2_error(predict: ms.Tensor, label: ms.Tensor):
    r"""
    calculate the Relative Root Mean Square Error.
        math:
            error = \sqrt{\frac{\sum_{i=1}^{N}{(x_i-y_i)^2}}{sum_{i=1}^{N}{(y_i)^2}}}
    :param predict: (batch_size, dim + dim)
    :param label: (batch_size, dim + dim)
    :return: (batch_size, )
    """
    batch_size = predict.shape[0]
    predict = predict.reshape(batch_size, -1)
    label = label.reshape(batch_size, -1)
    diff_norms = ms.ops.square(predict - label).sum(axis=1)
    label_norms = ms.ops.square(label).sum(axis=1)
    rel_error = ms.ops.sqrt(diff_norms) / ms.ops.sqrt(label_norms)
    return rel_error


def test_forward(model, test_dataset):
    error_list = []
    idx = 0
    for H, trajectories, time_points in test_dataset.fetch():
        former = trajectories[:, 0]
        error_t_list = []
        idx += 1
        print('H {} test forward ...'.format(idx))
        for t_idx in range(len(time_points) - 1):
            T = time_points[t_idx + 1] - time_points[t_idx]
            latter = model.construct(H=H, S0=former, T=T)
            rel_error = relative_l2_error(predict=latter, label=trajectories[:, t_idx + 1])
            error_t_list.append(rel_error.asnumpy())
            former = latter
        error_t_list = np.stack(error_t_list, axis=1)
        error_list.append(error_t_list)
    # data_num * time_num
    error_list = np.concatenate(error_list, axis=0)
    return error_list


def evaluate(config, model, data_generator: Generator):
    h_num = config['evaluate_h_num']
    init_num = config['evaluate_init_num']
    test_dataset = generate_test_data(data_generator=data_generator, config=config, h_num=h_num, init_num=init_num,
                                      flag='evaluate')
    errors = test_forward(model=model, test_dataset=test_dataset)
    max_error = np.max(errors)
    print('=================================== evaluate ===================================')
    print('-- Max evaluate relative L2 error: {}'.format(max_error))
    return


def load_param_dict(save_directory: str, step: int):
    ckpt_path = os.path.join(save_directory, 'dense_ode_net_step{}.ckpt'.format(step))
    param_dict = load_checkpoint(ckpt_file_name=ckpt_path)
    return param_dict


def show_test_acc(config, show_results: bool):
    acc = np.loadtxt(config['test_acc_path'], delimiter='\t')
    max_acc = np.max(acc, axis=0)
    min_acc = np.min(acc, axis=0)
    max_t = config['test_max_t']
    t_num = config['test_t_points_num']
    dt = max_t / t_num
    time_points = np.arange(dt, dt * (t_num + 1), dt)
    image_fig = os.path.join(config['images_directory'], 'accuracies.png')
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, min_acc, color='tomato')
    plt.plot(time_points, max_acc, color='tomato')
    plt.fill_between(time_points, min_acc, max_acc, alpha=0.5, color='tomato')
    acc_bot = np.min(min_acc)
    tick_bot = 100 - 1.5 * (100 - acc_bot)
    tick_interval = (100 - tick_bot) / 5
    y_ticks = np.arange(tick_bot, 100 + tick_interval, tick_interval)
    y_labels = ['{:.5f}'.format(y_ticks[i]) for i in range(y_ticks.shape[0])]
    plt.ylim(tick_bot, 100)
    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.title('Test acc = (1 - relative l2 error) * 100')
    plt.xlabel('Time')
    plt.ylabel('Test Accuracy %')
    plt.savefig(image_fig)
    print('============ Test acc image saved as {} ============'.format(image_fig))
    if show_results:
        plt.show()
    return


def test(config, show_results: bool = False):
    h_num = config['test_h_num']
    init_num = config['test_init_num']
    data_generator = Generator(dim=config['h_dim'])
    model = init_model(config)
    param_dict = load_param_dict(save_directory=config['save_directory'], step=config['time_step_num'])
    param_not_load = load_param_into_net(net=model, parameter_dict=param_dict)
    if len(param_not_load) == 0:
        print('=============== Net saved at step {} is loaded. ==============='.format(config['time_step_num']))
    else:
        print('!!!!!!!!! param not loaded: ', param_not_load)
    test_dataset = generate_test_data(data_generator=data_generator, config=config, h_num=h_num, init_num=init_num,
                                      flag='test')
    errors = test_forward(model=model, test_dataset=test_dataset)
    acc = (1 - errors) * 100
    np.savetxt(fname=config['test_acc_path'], X=acc, delimiter='\t')
    show_test_acc(config=config, show_results=show_results)
    return
