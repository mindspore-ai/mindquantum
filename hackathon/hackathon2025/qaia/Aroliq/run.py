import os
import time
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

Variables = Tuple[int, int, bool]   # [theta, nq_alpha, is_opt_amp]
Phases = np.ndarray
Amplitudes = np.ndarray

if 'import real answer implementation':
    from pathlib import Path
    BASE_PATH = Path(__file__).parent
    answer_suffixes = [fp.stem[len('answer'):] for fp in BASE_PATH.iterdir() if fp.suffix == '.py' and fp.stem.startswith('answer')]
    METHODS = [meth[1:] if meth.startswith('_') else meth for meth in answer_suffixes if meth]

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-M', '--method', choices=METHODS, help='run answer method')
    args, _ = parser.parse_known_args()
    method = args.method

    print('>> run method:', method or 'submit')
    if method:
        from importlib import import_module
        optimized = import_module(f'answer_{method}').optimized
    else:
        from answer import optimized


# 请参赛选手不要修改此文件
def get_score(phase_angle:Phases, amplitude:Amplitudes, variables:Variables) -> float:
    """
    打分函数,请参赛选手不要修改此函数和其中调用的任何函数
    Args:
        phase_angle 相位角，32个元素的列表float，取值为0到2\pi的弧度制
        amplitude  阵子振幅，32个元素的列表float
        variables  参数变量列表，变量列表中包含3个元素
            第一个元素是 theta_0 信号方向，float数
            第二个元素是 相位量化比特数，取值为 1, 2, 3, 4
            第三个元素是 控制振幅是否优化，取值为True 或 False
    Returns：
        单个角度对应优化参数的分数
    """

    n_angle = 500
    N = 32
    theta_0 = variables[0]
    n_bit_phase = variables[1]
    opt_amp_or_not = variables[2]

    # 确保相位和振幅是按照赛题要求的取值
    if opt_amp_or_not is True:
        amplitude = amplitude / np.max(amplitude)
    else:
        amplitude = np.ones(N)
    phase_angle = np.angle(np.exp(1.0j * phase_angle)) + np.pi
    phase_angle = np.round(phase_angle / (2 * np.pi) * (2 ** n_bit_phase)) / (2 ** n_bit_phase) * (2 * np.pi)

    efield = get_efield(n_angle, N)
    theta_array = np.linspace(0, 180, 180 * n_angle + 1)
    amp_phase = []
    for i in range(N):
        amp_phase.append(amplitude[i] * np.exp(1.0j * phase_angle[i]))
    F = np.einsum('i, ij -> j', np.array(amp_phase), efield)
    FF = np.real(F.conj() * F)
    # 峰值移到0dB
    db_array = 10 * np.log10(FF / np.max(FF))

    # [0°,180°] 位置范围，把θ移到原点0
    x = theta_array - theta_0
    value_list = []
    for i in range(theta_array.shape[0]):
        # >30°外旁瓣区域
        if abs(x[i]) >= 30:
            # 须在峰值-15dB下，故+15后不应大于0
            value_list.append(db_array[i] + 15)
    # 外旁瓣惩罚
    a = max(np.max(value_list), 0)

    # 峰值位置
    target = np.max(db_array)
    for i in range(theta_array.shape[0]):
        if db_array[i] == target:
            max_index = i
            break
    
    # 主瓣两侧最近的压制到 -30dB 的位置
    theta_up = 180
    theta_down = 0
    # 主瓣两侧最近的局部最小值位置
    theta_min_up = 180
    theta_min_down = 0
    # 峰值位置与目标位置相差大于1°，直接没分
    if abs(theta_array[max_index] - theta_0) > 1:
        y = 0
        print(f'Incorrect beamforming direction: {theta_array[max_index]}, with target: {theta_0}')
        print(f'final score: {y}')
    else:
        for i in range(1, 10000):
            if db_array[i + max_index] <= -30:
                theta_up = theta_array[i + max_index]
                break

        for i in range(1, 10000):
            if db_array[-i + max_index] <= -30:
                theta_down = theta_array[-i + max_index]
                break

        for i in range(1, 10000):
            if (db_array[i + max_index] < db_array[i - 1 + max_index]) and (db_array[i + max_index] < db_array[i + 1 + max_index]):
                theta_min_up = theta_array[i + max_index]
                break

        for i in range(1, 10000):
            if (db_array[-i + max_index] < db_array[-i - 1 + max_index]) and (db_array[-i + max_index] < db_array[-i + 1 + max_index]):
                theta_min_down = theta_array[-i + max_index]
                break

        if theta_up == 180 or theta_down == 0:
            # 主瓣在这里意味着一个高出左右地平线 30dB 的尖峰
            y = 0
            print(f'Failed to identify expected mainlobe.')
            print(f'final score: {y}')
        elif theta_min_up < theta_up or theta_min_down > theta_down:
            # 主瓣须是单峰，这意味着左右极小值必在-30dB检测点之外，即 theta_min_down <= theta_down < peak < theta_up <= theta_min_up
            y = 0
            print('>> assert fail: theta_min_down <= theta_down < peak < theta_up <= theta_min_up')
            print(f'   {theta_min_down} <= {theta_down} < {theta_array[max_index]} < {theta_up} <= {theta_min_up}')
            print(f'The intensity of mainlobe did not decrease to -30 dB')
            print(f'final score: {y}')
        else:
            # 主瓣宽度
            W = theta_up - theta_down
            # 大于6°有惩罚
            b = max(W - 6, 0)

            value_list_2 = []
            for i in range(theta_array.shape[0]):
                # <30°内旁瓣区域 and >主瓣左右极小值外(的非主瓣)区域
                if abs(x[i]) <= 30 and (x[i] >= theta_min_up - theta_0 or x[i] <= theta_min_down - theta_0):
                    # 须在峰值-30dB下，故+30后不应大于0
                    value_list_2.append(db_array[i] + 30)
            # 外旁瓣惩罚
            c = np.max(value_list_2)

            y_sum = 1000 - 100 * a - 80 * b - 20 * c
            # 负分直接归为0分
            y = max(y_sum, 0)
            print(f'W = {W:.5f}, a = {a:.5f}, b = {b:.5f}, c = {c:.5f}; y_sum = {y_sum:.5f}, final score: y = {y:.5f}')
    return float(y)


def get_efield(n_angle:int=500, N:int=32):
    # Eq. 3~4
    theta = np.linspace(0, 180, 180 * n_angle + 1)  # [90001]
    x = 12 * ((theta - 90) / 90) ** 2               # [90001], 下凹双曲线
    E_dB = -1.0 * np.where(x < 30, x, 30)           # [90001], 上凸双曲线
    E_theta = 10 ** (E_dB / 10)                     # [90001], 钟形/山坡
    EF = E_theta ** 0.5                             # [90001], WHY??
    # Eq. 2
    phase_x = 1j * np.pi * np.cos(theta * np.pi / 180)
    AF = np.exp(phase_x[None, :] * np.arange(N)[:, None])   # [32, 90001]
    # Eq. 1
    efield = EF[None, ...] * AF                             # [32, 90001]
    return efield


def get_single_score(variables:Variables):
    phase_angle, amp = optimized(variables)
    return get_score(phase_angle, amp, variables)


def judgment(timeout=90, variables=None):
    scores = []
    records = []
    count_timeout = 0
    for variable in variables:
        single_score = 0
        with ProcessPoolExecutor() as executor:
            future = executor.submit(get_single_score, variable)
            try:
                single_score = future.result(timeout=timeout)
            except Exception:
                count_timeout += 1
                for process in multiprocessing.active_children():
                    process.terminate()
                    process.join()
        rec = list(variable)
        rec.append(single_score)
        records.append(rec)
        scores.append(single_score)
    print('scores:', scores)

    os.makedirs('./tmp', exist_ok=True)
    save_fp = './tmp/grid_result.json'
    print(f'>> save to {save_fp}')
    with open(save_fp, 'w', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write('\n')
    return np.mean(scores), count_timeout


def judgment_single_thread(timeout=90, variables=None):
    scores = []
    count_timeout = 0
    for i_sample, variable in enumerate(variables):
        ts_start = time.time()
        scores.append(get_single_score(variable))
        ts_cost = time.time() - ts_start
        print(f'[{i_sample}] ts_cost: {ts_cost}')
        if ts_cost > timeout:
            count_timeout += 1
    print('scores:', scores)
    return np.mean(scores), count_timeout


if __name__ == '__main__':
    # 单个参数超时时间
    single_param_timeout = 180 # 90
    # 参数列表
    if not 'grid':
        variable_list: List[Variables] = []
        for theta_0 in range(45, 135+1, 10):
            for nq in [2, 3, 4]:
                for opt_amp in [True, False]:
                    variable_list.append((theta_0, nq, opt_amp))
        print(f'>> grid_test {len(variable_list)} samples')
    else:
        variable_list: List[Variables] = [
            [112, 4, True], 
            [112, 3, True], 
            [112, 2, True], 
            [ 80, 4, False],
            [ 80, 3, False],
            [ 80, 2, False],
        ]

    judgment_fn = judgment_single_thread if os.getenv('DEBUG') else judgment
    start = time.time()
    score, timeout_param = judgment_fn(single_param_timeout, variable_list)
    end = time.time()
    print("timecost:", end - start)
    if timeout_param:
        print(f"failed: {timeout_param} / {len(variable_list)} timeout")
    print(f"score: {score:.4f}")
