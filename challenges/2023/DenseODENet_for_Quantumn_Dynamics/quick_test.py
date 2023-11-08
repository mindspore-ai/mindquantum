import argparse
from utils import init_env, makedir, get_config, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dense ODE-Net for schrodinger test")
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

    my_config['time_step_num'] = 20

    makedir(my_config)
    test(my_config)
